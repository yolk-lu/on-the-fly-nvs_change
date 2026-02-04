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
from scene.sg_utils import compute_sh_from_sg


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

        # Check for SG params
        sg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_")]
        sg_names = sorted(sg_names, key=lambda x: int(x.split("_")[-1]))
        is_sg = len(sg_names) > 0
        
        gaussian_params = {
            "xyz": {"val": torch.tensor(xyz, dtype=torch.float)},
            "f_dc": {"val": torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()},
            "scaling": {}, # Filled later
            "rotation": {}, # Filled later
            "opacity": {"val": torch.tensor(opacities, dtype=torch.float)},
        }

        if is_sg:
            # Load SG params
            # Expect N channels (e.g. 7 for 1 lobe)
            sg_data = np.zeros((xyz.shape[0], len(sg_names)))
            for idx, attr_name in enumerate(sg_names):
                sg_data[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            # Reshape to (N, 1, 7) assuming 1 lobe for now. 
            # If we had multiple lobes, we'd need to know the structure or infer from count.
            # Current implementation uses (N, 1, 7).
            sg_tensor = torch.tensor(sg_data, dtype=torch.float).unsqueeze(1)
            gaussian_params["sg_params"] = {"val": sg_tensor}
            
            # We can also load f_rest if present, but it's redundant/derived. 
            # We'll skip loading f_rest into gaussian_params to rely on on-the-fly generation.
        else:
            # Load SH params (Legacy)
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            # assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
            
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
            )
            f_rest_tensor = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
            gaussian_params["f_rest"] = {"val": f_rest_tensor}

        
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        gaussian_params["scaling"]["val"] = torch.tensor(scales, dtype=torch.float)

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        gaussian_params["rotation"]["val"] = torch.tensor(rots, dtype=torch.float)

        return cls(gaussian_params, position.cuda(), [])

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.gaussian_params["f_dc"]["val"].shape[2]):
            l.append("f_dc_{}".format(i))
        
        # Viewer Compatibility: Always save f_rest (Standard SH)
        # 15 coeffs * 3 channels = 45 attributes
        # Standard 3DGS stores them as f_rest_0 ... f_rest_44
        num_sh_rest = 45 # Degree 3
        for i in range(num_sh_rest):
            l.append("f_rest_{}".format(i))
            
        # If SG Params exist, save them as custom attributes for training resumption
        if "sg_params" in self.gaussian_params:
            sg_val = self.gaussian_params["sg_params"]["val"]
            # Flatten: (N, 1, 7) -> (N, 7) -> 7 attributes
            num_sg = sg_val.shape[1] * sg_val.shape[2]
            for i in range(num_sg):
                l.append("sg_{}".format(i))
        
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
        
        # Prepare f_rest (SH) and sg_data (SG)
        f_rest = None
        sg_data = None
        
        if "sg_params" in self.gaussian_params:
            # SG Mode: Convert to SH for viewer compatibility
            sg_val = self.gaussian_params["sg_params"]["val"].detach() # (N, 1, 7)
            
            amplitude = torch.exp(sg_val[..., 0:3])
            axis = sg_val[..., 3:6]
            sharpness = torch.exp(sg_val[..., 6:7])
            
            # Compute SH (Degree 3)
            _, f_rest_tensor = compute_sh_from_sg(amplitude, axis, sharpness, degree=3)
            # f_rest_tensor: (N, 15, 3)
            
            f_rest = to_numpy(
                f_rest_tensor
                .transpose(1, 2) # (N, 3, 15)
                .flatten(start_dim=1) # (N, 45)
            )
            
            # Save SG Params as extra attributes
            sg_data = to_numpy(
                sg_val
                .flatten(start_dim=1) # (N, 7)
            )
            
        elif "f_rest" in self.gaussian_params:
            # Standard SH Mode
            f_rest = to_numpy(
                self.gaussian_params["f_rest"]["val"]
                .detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
            )
        else:
            # Fallback (Deg 0?)
            f_rest = np.zeros((xyz.shape[0], 45))

        opacities = to_numpy(self.gaussian_params["opacity"]["val"])
        scale = to_numpy(self.gaussian_params["scaling"]["val"])
        rotation = to_numpy(self.gaussian_params["rotation"]["val"])

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # Concatenate: [xyz, normals, f_dc, f_rest, (optional sg), opacities, scale, rot]
        # Order must match construct_list_of_attributes
        
        cat_list = [xyz, normals, f_dc, f_rest]
        if sg_data is not None:
            cat_list.append(sg_data)
        cat_list.extend([opacities, scale, rotation])
        
        attributes = np.concatenate(cat_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
