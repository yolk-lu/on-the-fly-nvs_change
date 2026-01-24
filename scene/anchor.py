#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations
import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

from scene.keyframe import Keyframe
from utils import inverse_sigmoid, to_numpy


class Anchor:
    """
    Represents an anchor that holds Gaussian parameters and associated keyframes.
    """
    def __init__(
        self,
        gaussian_params: dict[str, dict[str, torch.Tensor]],
        position: torch.Tensor = torch.zeros(3, dtype=torch.float32, device="cuda"),
        keyframes: list[Keyframe] = [],
    ):
        self.gaussian_params = gaussian_params
        self.position = position
        self.keyframes = keyframes
        self.keyframe_ids = [keyframe.index for keyframe in keyframes]

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        self.keyframe_ids.append(keyframe.index)

    def increase_lod(self, num_childs, scaling_lower_bound_old, scaling_lower_bound_new):
        new_params = {}
        for key, param in self.gaussian_params.items():
            val = param["val"]
            # Repeat the values to create children
            # valid for [N, ...] tensors
            # Use repeat_interleave to support Per-Gaussian split counts (Tensor argument)
            # and to keep children adjacent to parents ([A, A, B, B...])
            new_val = val.repeat_interleave(num_childs, dim=0)
            
            if key == "scaling":
                # Adjust scaling to preserve effective size
                # Linear scale = exp(log_scale) + lower_bound
                parent_linear_scale = torch.exp(new_val) + scaling_lower_bound_old
                # New log scale = log(parent_linear_scale - new_lower_bound)
                # Ensure positive argument for log
                arg = parent_linear_scale - scaling_lower_bound_new
                arg = torch.clamp(arg, min=1e-6)
                new_val = torch.log(arg)
            
            # new_params[key] = {val: new_val}

            # Detach to ensure these are new leaf nodes for the optimizer
            new_params[key] = {"val": new_val.detach().requires_grad_(True)}
            # Copy other keys if any (like lr)
            for k, v in param.items():
                if k == "val":
                    continue
                
                # Handle Learning Rate (lr)
                # If it's a tensor matching the number of particles, we must duplicate it
                if k == "lr" and isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == val.shape[0]:
                    new_v = v.repeat_interleave(num_childs, dim=0)
                    new_params[key][k] = new_v
                    continue
                    
                # Skip optimizer states (exp_avg, exp_avg_sq) to force re-initialization
                if k in ["exp_avg", "exp_avg_sq"]:
                    continue

                # Default copy for other scalar/config values
                new_params[key][k] = v
                    
        self.gaussian_params = new_params


    def duplicate_param_dict(self):
        self.gaussian_params = {
            key: {k: v for k, v in value.items()}
            for key, value in self.gaussian_params.items()
        }

    @property
    def device(self):
        return self.gaussian_params["xyz"]["val"].device

    def to(self, device, with_keyframes=False):
        if self.device != device:
            for param in self.gaussian_params.values():
                for key, tensor in param.items():
                    if type(param[key]) is torch.Tensor:
                        param[key] = tensor.to(device)
            if with_keyframes:
                for keyframe in self.keyframes:
                    keyframe.to(device)
        return self

    @classmethod
    @torch.no_grad()
    def blend(
        cls, cam_centre: torch.Tensor, anchors: list[Anchor], anchor_overlap: float
    ) -> tuple[dict[str, dict[str, torch.Tensor]], np.ndarray]:
        """
        Blend the Gaussian parameters of the closest anchors based on their distance to the centre of the camera to render.
        """
        anchor_weights = np.zeros(len(anchors))
        anchor_positions = torch.stack(
            [anchor.position for anchor in anchors], dim=0
        )
        anchor_dists = torch.linalg.vector_norm(
            anchor_positions - cam_centre[None], dim=-1
        )
        closest_anchors_dist, closest_anchors_ids = torch.topk(
            anchor_dists, min(3, len(anchors)), largest=False
        )
        ratio = (
            (closest_anchors_dist[0]) / (closest_anchors_dist[1])
            if len(anchors) > 1
            else 0
        )

        for anchor_id in range(len(anchors)):
            if anchor_id in closest_anchors_ids:
                anchors[anchor_id].to("cuda")
            else:
                anchors[anchor_id].to("cpu")

        # Apply eq. 5
        if ratio < (1 - anchor_overlap):
            gaussian_params = anchors[closest_anchors_ids[0]].gaussian_params
            anchor_weights[closest_anchors_ids[0]] = 1
        else:
            # Blend the opacities of the two closest anchors
            blending_weights = 1 - (ratio - (1 - anchor_overlap)) * (
                0.5 / anchor_overlap
            )
            params1 = anchors[closest_anchors_ids[0]].gaussian_params
            params2 = anchors[closest_anchors_ids[1]].gaussian_params
            gaussian_params = {
                name: {"val": torch.cat([params1[name]["val"], params2[name]["val"]], dim=0)}
                for name in params1
                if name != "opacity"
            }
            gaussian_params["opacity"] = {
                "val": torch.cat(
                    [
                        inverse_sigmoid(torch.sigmoid(params1["opacity"]["val"]) * blending_weights),
                        inverse_sigmoid(torch.sigmoid(params2["opacity"]["val"]) * (1 - blending_weights)),
                    ],
                    dim=0,
                )
            }

            # Used for visualization
            anchor_weights[closest_anchors_ids[0]] = blending_weights
            anchor_weights[closest_anchors_ids[1]] = 1 - blending_weights

        return gaussian_params, anchor_weights

    @classmethod
    def from_ply(cls, anchor_path: str, position: torch.Tensor, max_sh_degree: str):
        plydata = PlyData.read(anchor_path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        
        # Check for SG mode (7 features) vs SH mode
        is_sg = len(extra_f_names) == 7
        if not is_sg:
            assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if is_sg:
            # SG mode: (N, 7) directly, no transpose needed usually or handled in init
            # Tensor expects (N, 7) based on SceneModel init
            f_rest_tensor = torch.tensor(features_extra, dtype=torch.float)
        else:
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
            )
            f_rest_tensor = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()

        # ... (rest of loading) ...
        
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussian_params = {
            "xyz": {"val": torch.tensor(xyz, dtype=torch.float)},
            "f_dc": {"val": torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()},
            "f_rest": {"val": f_rest_tensor},
            "scaling": {"val": torch.tensor(scales, dtype=torch.float)},
            "rotation": {"val": torch.tensor(rots, dtype=torch.float)},
            "opacity": {"val": torch.tensor(opacities, dtype=torch.float)},
        }

        return cls(gaussian_params, position.cuda(), [])

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.gaussian_params["f_dc"]["val"].shape[2]):
            l.append("f_dc_{}".format(i))
        
        #---------------------------------------------------------------------------
        # TODO: fix for SG dimensions

        f_rest_val = self.gaussian_params["f_rest"]["val"]
        if f_rest_val.dim() == 2:
            # SG 模式: (N, 7)，特徵數就是第 1 維的大小
            num_rest_features = f_rest_val.shape[1]
        else:
            # SH 模式: (N, 15, 3)，特徵數是 15 * 3
            num_rest_features = f_rest_val.shape[1] * f_rest_val.shape[2]
            
        for i in range(num_rest_features):
            l.append("f_rest_{}".format(i))
        
        #--------------------------------------------------------------------------
        # for i in range(
        #     self.gaussian_params["f_rest"]["val"].shape[1]
        #     * self.gaussian_params["f_rest"]["val"].shape[2]
        # ):
        #     l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.gaussian_params["scaling"]["val"].shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.gaussian_params["rotation"]["val"].shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = to_numpy(self.gaussian_params["xyz"]["val"])
        normals = np.zeros_like(xyz)
        f_dc = to_numpy(
            self.gaussian_params["f_dc"]["val"]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
        )
        # TODO: fix for SG dimensions

        f_rest_tensor = self.gaussian_params["f_rest"]["val"].detach()
        
        if f_rest_tensor.dim() == 3:
            # 舊的 SH 模式: (N, 15, 3) -> 需要 transpose(1, 2)
            f_rest = to_numpy(
                f_rest_tensor
                .transpose(1, 2)
                .flatten(start_dim=1)
            )
        else:
            # 新的 SG 模式: (N, 7) -> 直接轉 Numpy，不需要 transpose
            f_rest = to_numpy(f_rest_tensor)



        # f_rest = to_numpy(
        #     self.gaussian_params["f_rest"]["val"]
        #     .detach()
        #     .transpose(1, 2)
        #     .flatten(start_dim=1)
        # )


        opacities = to_numpy(self.gaussian_params["opacity"]["val"])
        scale = to_numpy(self.gaussian_params["scaling"]["val"])
        rotation = to_numpy(self.gaussian_params["rotation"]["val"])

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
