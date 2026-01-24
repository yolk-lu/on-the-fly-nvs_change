
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

def load_scene(args, lod):
    print(f"Loading scene from {args.model_path} with LoD={lod}")
    scene = SceneModel.from_scene(args.model_path, args, lod=lod)
    return scene

def render_frame_dual_lod(scene_coarse, scene_fine, render_camera, args):
    # This function is not currently used in the main loop but kept for reference
    pass 

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
    
    # Zoom Control
    parser.add_argument('--zoom_mult', type=float, default=1.0, help="Zoom Multiplier (e.g. 2.0 = 2x Zoom, 0.5 = Wide Angle)")

    args = parser.parse_args()
    
    tracker = ResourceTracker()

    with tracker.track("Load Scenes"):
        # Load Fine Model
        print(f"Loading Fine Scene from {args.model_path}")
        scene_fine = SceneModel.from_scene(args.model_path, args, lod=args.lod_fine)
        
        # Load Coarse Model (if different)
        if args.lod_coarse != args.lod_fine:
            # Deduce Coarse Path
            # Assume standard structure: .../lod_X
            if f"lod_{args.lod_fine}" in args.model_path:
                coarse_path = args.model_path.replace(f"lod_{args.lod_fine}", f"lod_{args.lod_coarse}")
            else:
                # Fallback: Assume sibling directory
                coarse_path = os.path.join(os.path.dirname(args.model_path.rstrip("/")), f"lod_{args.lod_coarse}")
            
            if not os.path.exists(coarse_path):
                print(f"WARNING: Coarse path {coarse_path} not found! Trying to load from Fine path...")
                coarse_path = args.model_path
            
            print(f"Loading Coarse Scene from {coarse_path}")
            scene_coarse = SceneModel.from_scene(coarse_path, args, lod=args.lod_coarse)

    # Configure Perceptual LoD
    scene_fine.perceptual_lod_config = {
        'k_ecc': args.k_ecc,
        'tau_min': args.tau_min,
        'tau_max': args.tau_max
    }
    
    # Enable Inference Mode (Important for Anchors)
    scene_fine.inference_mode = True
    if 'scene_coarse' in locals():
        scene_coarse.inference_mode = True
    
    with tracker.track("Load Path"):
        # Read and parse the render path
        if args.render_path.endswith('.bin') or args.render_path.endswith('.txt'):
            # It's a colmap model
            render_cameras, render_images, _ = read_model(args.render_path)
        else:
            # Assume it's a directory containing colmap model
            render_cameras, render_images, _ = read_model(args.render_path)

        # Read the path used to align the render path to the scene
        if args.alignment_path != "":
            scene_images_names = [
                keyframe.info["name"] for keyframe in scene_fine.keyframes
            ]
            scene_cam_centers = torch.stack(
                [keyframe.get_centre() for keyframe in scene_fine.keyframes]
            ).to("cuda")
            scene_Rts = torch.stack(
                [keyframe.get_Rt() for keyframe in scene_fine.keyframes]
            ).to("cuda")
            _, alignment_images, _ = read_model(args.alignment_path)

            # only keep input poses that are in the scene keyframes
            alignment_extrinsics_dict = {
                os.path.basename(extr.name): extr for extr in alignment_images.values()
            }
            alignment_extrinsics = [
                alignment_extrinsics_dict[name]
                for name in scene_images_names
                if name in alignment_extrinsics_dict
            ]

            alignment_Rts = torch.eye(4).to("cuda").repeat(len(alignment_extrinsics), 1, 1)
            alignment_cam_centers = torch.zeros((len(alignment_extrinsics), 3)).to("cuda")
            for idx, cam_extrinsics in enumerate(alignment_extrinsics):
                alignment_Rts[idx][:3, :3] = torch.Tensor(qvec2rotmat(cam_extrinsics.qvec))
                alignment_Rts[idx][:3, 3] = torch.Tensor(cam_extrinsics.tvec)
                alignment_cam_centers[idx] = -alignment_Rts[idx][:3, :3].T @ alignment_Rts[idx][:3, 3]


    os.makedirs(args.out_dir, exist_ok=True)
    render_camera = render_cameras[list(render_cameras.keys())[0]]
    out = cv2.VideoWriter(
        os.path.join(args.out_dir, f"rendered_path_smooth_zoom{args.zoom_mult}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.framerate,
        (render_camera.width, render_camera.height),
        True,
    )

    print(f"Rendering frames (Zoom = {args.zoom_mult})")
    
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

        # Align the camera pose to the scene
        if args.alignment_path != "":
            # Get the training cameras closest to the camera to render
            n_closest = 10
            Rt_inv = torch.linalg.inv(Rt)
            path_cam_center = Rt_inv[:3, 3]
            distances = torch.linalg.norm(
                path_cam_center[None] - alignment_cam_centers, dim=1
            )
            closest_indices = distances.topk(n_closest, largest=False).indices

            # Get the transform that aligns the alignment cameras to the scene
            R, t, s = get_transform_mean_up_fwd(
                alignment_Rts[closest_indices], scene_Rts[closest_indices], True
            )

            # Apply the transform to the camera to render
            Rt_inv[:3, :3] = R @ Rt_inv[:3, :3]
            Rt_inv[:3, 3] = R @ Rt_inv[:3, 3] * s + t
            Rt = torch.linalg.inv(Rt_inv)

        # Render
        # 1. Render Coarse (as Background)
        bg = torch.zeros(3, device="cuda")
        if 'scene_coarse' in locals():
            with tracker.track("Render Coarse"):
                with torch.no_grad():
                    # Apply same perceptual config to coarse?
                    # Yes, so it also fades if it gets extremely small/far
                    scene_coarse.perceptual_lod_config = scene_fine.perceptual_lod_config
                    
                    render_dict_coarse = scene_coarse.render(
                        width=render_camera.width,
                        height=render_camera.height,
                        fov_x=fov_x,
                        fov_y=fov_y,
                        view_matrix=(Rt.transpose(0, 1)).to("cuda"),
                        scaling_modifier=1.0,
                        bg=bg # Black base
                    )
                    bg = render_dict_coarse["render"]
                
        # 2. Render Fine (on top of Coarse)
        with tracker.track("Render Fine"):
            with torch.no_grad():
                render_dict = scene_fine.render(
                    width=render_camera.width,
                    height=render_camera.height,
                    fov_x=fov_x,
                    fov_y=fov_y,
                    view_matrix=(Rt.transpose(0, 1)).to("cuda"),
                    scaling_modifier=1.0,
                    bg=bg # Use Coarse result as BG
                )
                render = render_dict["render"].clamp(0, 1.0)
        
        with tracker.track("Save Frame"):
            frame = render.mul(255).permute(1, 2, 0).byte().cpu().numpy()[:, :, ::-1]
            out.write(frame)
            
            # Optional: Save individual frames
            cv2.imwrite(os.path.join(args.out_dir, f"{render_image.name}"), frame)

    out.release()
    
    # Save statistics
    stats_path = os.path.join(args.out_dir, "resource_stats.txt")
    print(f"Saving resource stats to {stats_path}")
    tracker.save_stats(stats_path)
    tracker.print_stats()
