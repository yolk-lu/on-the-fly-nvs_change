
import argparse
import sys
import os
import math
sys.path.append(".")
from dataloaders.read_write_model import read_model, qvec2rotmat
from scene.scene_model import SceneModel
from utils import focal2fov, get_transform_mean_up_fwd
import torch
from tqdm import tqdm
import cv2
from resource_tracker import ResourceTracker
import numpy as np

def load_scene(args, lod):
    print(f"Loading scene from {args.model_path} with LoD={lod}")
    scene = SceneModel.from_scene(args.model_path, args, lod=lod)
    return scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--lod_fine", type=int, default=3, help="Fine LoD Level")
    parser.add_argument("--lod_coarse", type=int, default=1, help="Coarse LoD Level (Optional)")
    parser.add_argument("--render_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alignment_path", default="")
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--anchor_overlap", type=float, default=0.3, help="Anchor overlap ratio")
    
    # Perceptual params
    parser.add_argument('--k_ecc', type=float, default=0.5)
    parser.add_argument('--tau_min', type=float, default=1.0)
    parser.add_argument('--tau_max', type=float, default=5.0)
    
    # Zoom/Mixing Control
    parser.add_argument('--zoom_mult', type=float, default=1.0, help="Zoom Multiplier")
    parser.add_argument('--dist_threshold', type=float, default=5.0, help="Distance threshold for switching to Coarse LoD")

    args = parser.parse_args()
    
    tracker = ResourceTracker()

    with tracker.track("Load Scenes"):
        # Load Fine Model
        print(f"Loading Fine Scene from {args.model_path}")
        scene_fine = SceneModel.from_scene(args.model_path, args, lod=args.lod_fine)
        
        # Load Coarse Model (if different)
        if args.lod_coarse != args.lod_fine:
            # Deduce Coarse Path
            if f"lod_{args.lod_fine}" in args.model_path:
                coarse_path = args.model_path.replace(f"lod_{args.lod_fine}", f"lod_{args.lod_coarse}")
            else:
                coarse_path = os.path.join(os.path.dirname(args.model_path.rstrip("/")), f"lod_{args.lod_coarse}")
            
            if not os.path.exists(coarse_path):
                print(f"WARNING: Coarse path {coarse_path} not found! Trying to load from Fine path...")
                coarse_path = args.model_path
            
            print(f"Loading Coarse Scene from {coarse_path}")
            scene_coarse = SceneModel.from_scene(coarse_path, args, lod=args.lod_coarse)
        else:
            scene_coarse = scene_fine

    # Configure Perceptual LoD
    scene_fine.perceptual_lod_config = {
        'k_ecc': args.k_ecc,
        'tau_min': args.tau_min,
        'tau_max': args.tau_max
    }
    
    # Enable Inference Mode
    scene_fine.inference_mode = True
    scene_coarse.inference_mode = True
    
    with tracker.track("Load Path"):
        # Read and parse the render path
        if args.render_path.endswith('.bin') or args.render_path.endswith('.txt'):
            render_cameras, render_images, _ = read_model(args.render_path)
        else:
            render_cameras, render_images, _ = read_model(args.render_path)
        
        # NOTE: Alignment logic skipped for brevity/cleanliness, assumed not critical for this demo script
        # if needed, insert standard alignment block here

    os.makedirs(args.out_dir, exist_ok=True)
    render_camera = render_cameras[list(render_cameras.keys())[0]]
    out = cv2.VideoWriter(
        os.path.join(args.out_dir, f"rendered_path_mixed_dist{args.dist_threshold}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.framerate,
        (render_camera.width, render_camera.height),
        True,
    )

    print(f"Rendering frames (Mix Threshold = {args.dist_threshold})")
    
    for render_image in tqdm(render_images.values()):
        render_camera = render_cameras[render_image.camera_id]
        
        # Calculate Base FOV
        fov_x_base = focal2fov(render_camera.params[0], render_camera.width)
        fov_y_base = focal2fov(render_camera.params[0], render_camera.height)
        
        # Apply Zoom
        tan_fovx_new = math.tan(fov_x_base / 2.0) / args.zoom_mult
        tan_fovy_new = math.tan(fov_y_base / 2.0) / args.zoom_mult
        
        fov_x = 2.0 * math.atan(tan_fovx_new)
        fov_y = 2.0 * math.atan(tan_fovy_new)

        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = torch.Tensor(qvec2rotmat(render_image.qvec)).to("cuda")
        Rt[:3, 3] = torch.Tensor(render_image.tvec).to("cuda")

        view_matrix = Rt.transpose(0, 1)

        # 1. Prepare Parameters (Blend Anchors)
        scene_coarse.prepare_for_render(view_matrix)
        scene_fine.prepare_for_render(view_matrix)
        
        # 2. Get Raw Tensors
        xyz_c = scene_coarse.gaussian_params["xyz"]["val"]
        xyz_f = scene_fine.gaussian_params["xyz"]["val"]
        
        # 3. Compute Distance to Camera (from Coarse Centers)
        cam_pos = Rt[:3, 3]
        dists = torch.norm(xyz_c - cam_pos, dim=1)
        
        # 4. Generate Masks
        # Near/Detailed -> Use Fine
        # Far/Coarse -> Use Coarse
        mask_fine_selector = dists < args.dist_threshold
        mask_coarse_selector = ~mask_fine_selector
        
        # 5. Handle Ratio expansion
        if xyz_f.shape[0] % xyz_c.shape[0] == 0:
            ratio = xyz_f.shape[0] // xyz_c.shape[0]
            # Expand mask for fine
            mask_fine_expanded = mask_fine_selector.repeat_interleave(ratio)
            
            # 6. Extract Subsets
            # active_coarse_mask = mask_coarse_selector # Use boolean for coarse directly
        else:
            # Fallback (Mismatch topology): Use Fine everywhere
            # print("Topology mismatch! Using Fine model only.")
            mask_fine_expanded = torch.ones_like(xyz_f[:,0], dtype=torch.bool)
            mask_coarse_selector = torch.zeros_like(xyz_c[:,0], dtype=torch.bool)

        # 7. Construct Mixed Parameters
        mixed_params = {}
        # Iterate keys available in Fine check
        # Note: keys must exist in both. Assuming consistent schema.
        for key in scene_fine.gaussian_params.keys():
            val_c = scene_coarse.gaussian_params[key]["val"]
            val_f = scene_fine.gaussian_params[key]["val"]
            
            subset_c = val_c[mask_coarse_selector]
            subset_f = val_f[mask_fine_expanded]
            
            mixed_params[key] = torch.cat([subset_c, subset_f], dim=0)

        # 8. Render with Override
        with tracker.track("Render Mixed"):
            with torch.no_grad():
                render_dict = scene_fine.render(
                    width=render_camera.width,
                    height=render_camera.height,
                    fov_x=fov_x,
                    fov_y=fov_y,
                    view_matrix=view_matrix,
                    scaling_modifier=1.0,
                    bg=torch.zeros(3, device="cuda"),
                    override_params=mixed_params
                )
                render = render_dict["render"].clamp(0, 1.0)
        
        with tracker.track("Save Frame"):
            frame = render.mul(255).permute(1, 2, 0).byte().cpu().numpy()[:, :, ::-1]
            out.write(frame)

    out.release()
    
    # Save statistics
    stats_path = os.path.join(args.out_dir, "resource_stats.txt")
    print(f"Saving resource stats to {stats_path}")
    tracker.save_stats(stats_path)
    tracker.print_stats()
