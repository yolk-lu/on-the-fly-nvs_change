
# python scripts/render_lod.py -m ./results/Mip-NeRF360/garden/downsample_4/lod_1 \
#         --lod 1 \
#         --render_path /home/cglab/project/on-the-fly-nvs/results/Mip-NeRF360/garden/downsample_4/lod_1/colmap \
#         --out_dir ./results/Mip-NeRF360/garden/downsample_4/lod_1/render_images \

# python scripts/render_path.py -m ./results/Mip-NeRF360/garden/downsample_8/ \
#         --render_path /home/cglab/project/on-the-fly-nvs/results/Mip-NeRF360/garden/downsample_8/colmap \
#         --out_dir ./results/Mip-NeRF360/garden/downsample_8/render_images \


# python scripts/render_lod.py -m ./results/Mip-NeRF360/garden/downsample_8/lod_1 \
#         --lod 1 \
#         --render_path /home/cglab/project/on-the-fly-nvs/results/Mip-NeRF360/garden/downsample_8/lod_1/colmap \
#         --out_dir ./results/Mip-NeRF360/garden/downsample_8/lod_1/render_images \

# python scripts/render_lod.py -m ./results/Mip-NeRF360/garden/downsample_8/lod_2 \
#         --lod 2 \
#         --render_path /home/cglab/project/on-the-fly-nvs/results/Mip-NeRF360/garden/downsample_8/lod_2/colmap \
#         --out_dir ./results/Mip-NeRF360/garden/downsample_8/lod_2/render_images \

# python scripts/render_lod.py -m ./results/Mip-NeRF360/garden/downsample_8/lod_5 \
#         --lod 5 \
#         --render_path /home/cglab/project/on-the-fly-nvs/results/Mip-NeRF360/garden/downsample_8/lod_5/colmap \
#         --out_dir ./results/Mip-NeRF360/garden/downsample_8/lod_5/render_images \

# python scripts/render_smooth_lod.py \
#     -m ./results/Mip-NeRF360/garden/downsample_8/lod_3 \
#     --lod 3 \
#     --render_path ./results/Mip-NeRF360/garden/downsample_8/colmap \
#     --out_dir ./results/Mip-NeRF360/garden/downsample_8/lod_3/render_smooth \
#     --tau_min 0.8 --tau_max 2.0 \
#     --k_ecc 0.3

# python scripts/render_smooth_lod.py \
#     -m ./results/Mip-NeRF360/garden/downsample_8/lod_5 \
#     --lod_fine 5 \
#     --lod_coarse 1 \
#     --render_path ./results/Mip-NeRF360/garden/downsample_8/colmap \
#     --out_dir ./results/Mip-NeRF360/garden/downsample_8/lod_5/render_smooth \
#     --tau_min 0.8 --tau_max 2.0 \
#     --k_ecc 0.3



# Example: Dynamic LoD with 2x Zoom (Detail should appear in center)
# python scripts/render_smooth_lod.py \
#    -m ./results/Mip-NeRF360/garden/changing_LoD_v1/downsample_3.5/lod_3 \
#    --lod_fine 3 \
#    --lod_coarse 1 \
#    --render_path ./results/Mip-NeRF360/garden/changing_LoD_v1/downsample_3.5/colmap \
#    --out_dir ./results/Mip-NeRF360/garden/changing_LoD_v1/downsample_3.5/lod_3/render_zoom2x \
#    --tau_min 0.8 --tau_max 2.0 \
#    --k_ecc 0.3 \
#    --zoom_mult 2.0

# diff-gaussian-rasterizer-SH-Mix-SG test
python scripts/render_smooth_lod.py \
   -m ./results/Mip-NeRF360/garden/changing_LoD_v1/diff_gaussian_rasterizer_SH_MIX_SG_downsample3.5/lod_3 \
   --lod_fine 3 \
   --lod_coarse 1 \
   --render_path ./results/Mip-NeRF360/garden/changing_LoD_v1/diff_gaussian_rasterizer_SH_MIX_SG_downsample3.5/colmap \
   --out_dir ./results/Mip-NeRF360/garden/changing_LoD_v1/diff_gaussian_rasterizer_SH_MIX_SG_downsample3.5/render_SH_MIX_SG \
   --tau_min 0.8 --tau_max 2.0 \
   --k_ecc 0.3 \
