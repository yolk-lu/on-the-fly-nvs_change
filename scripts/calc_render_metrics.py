
import argparse
import os
import sys
import torch
import cv2
import lpips
from tqdm import tqdm
from fused_ssim import fused_ssim

sys.path.append(".")
from utils import psnr

def calculate_metrics(render_dir, gt_dir, stride=1, device="cuda"):
    print(f"Calculating metrics between:")
    print(f"Render: {render_dir}")
    print(f"GT:     {gt_dir}")

    if not os.path.exists(render_dir):
        print(f"Error: Render directory {render_dir} does not exist.")
        return
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory {gt_dir} does not exist.")
        return

    # LPIPS model
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    # Get images
    render_images = sorted([f for f in os.listdir(render_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gt_images = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Filter by name matching
    # Often render names match GT names. 
    # If not, we might need a mapping or strict intersection.
    common_names = sorted(list(set(render_images) & set(gt_images)))
    
    if len(common_names) == 0:
        print("Warning: No common image filenames found between directories.")
        # Fallback: simple sorting match if counts equal?
        if len(render_images) == len(gt_images) and len(render_images) > 0:
            print("Trying to match by index since filenames differ...")
            pairs = list(zip(render_images, gt_images))
        else:
            print("Aborting.")
            return
    else:
        pairs = [(f, f) for f in common_names]

    # Apply stride
    if stride > 1:
        print(f"Applying stride={stride}. Using {len(pairs[::stride])} of {len(pairs)} images.")
        pairs = pairs[::stride]

    print(f"Found {len(pairs)} image pairs.")

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    count = 0

    import numpy as np

    for name_ren, name_gt in tqdm(pairs):
        path_ren = os.path.join(render_dir, name_ren)
        path_gt = os.path.join(gt_dir, name_gt)

        image = cv2.imread(path_ren)
        gt_image = cv2.imread(path_gt)
        
        if image is None or gt_image is None:
            print(f"Warning: Could not read {name_ren} or {name_gt}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Resize render to GT if needed (or vice versa, usually render matches GT)
        if image.shape != gt_image.shape:
             image = cv2.resize(image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_AREA)

        # To Tensor
        image_t = torch.from_numpy(image).permute(2, 0, 1).to(device).float() / 255.0
        gt_image_t = torch.from_numpy(gt_image).permute(2, 0, 1).to(device).float() / 255.0

        # Calculate metrics
        # LPIPS expects NCHW
        with torch.no_grad():
            curr_lpips = lpips_model(image_t.unsqueeze(0), gt_image_t.unsqueeze(0)).item()
            curr_ssim = fused_ssim(image_t.unsqueeze(0), gt_image_t.unsqueeze(0), train=False).item()
            curr_psnr = psnr(image_t, gt_image_t).item() # psnr util often takes CHW or NCHW? Check util.
            # Checking utils.py: psnr(img1, img2) usually assumes same shape. 
            # If utils.psnr returns tensor, .item() is needed.

        if count % 5 == 0:
            torch.cuda.empty_cache()

        avg_psnr += curr_psnr
        avg_ssim += curr_ssim
        avg_lpips += curr_lpips
        count += 1
        
        # Explicitly delete tensors to free memory immediately
        del image_t
        del gt_image_t
        del image
        del gt_image

    if count > 0:
        avg_psnr /= count
        avg_ssim /= count
        avg_lpips /= count

        print("\n" + "="*40)
        print(f"Evaluation Results ({count} images)")
        print("-" * 40)
        print(f"PSNR:  {avg_psnr:.4f}")
        print(f"SSIM:  {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")
        print("="*40 + "\n")
        
        # Save to file
        out_path = os.path.join(render_dir, "metrics.txt")
        with open(out_path, "w") as f:
             f.write(f"PSNR: {avg_psnr:.4f}\n")
             f.write(f"SSIM: {avg_ssim:.4f}\n")
             f.write(f"LPIPS: {avg_lpips:.4f}\n")
        print(f"Metrics saved to {out_path}")

    else:
        print("No valid pairs processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", required=True, help="Directory containing rendered images")
    parser.add_argument("--gt_dir", required=True, help="Directory containing Ground Truth images")
    parser.add_argument("--stride", type=int, default=1, help="Calculate metrics for every N-th image (default: 1)")
    args = parser.parse_args()

    calculate_metrics(args.render_dir, args.gt_dir, stride=args.stride)
