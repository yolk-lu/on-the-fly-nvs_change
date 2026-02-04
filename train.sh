

# python train.py -s /home/cglab/project/Dataset_opensource/Drone-dataset/chiayi_0326 -m results/Drone/Chiayi_0326 


# python train.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/downsample_4 --downsampling 4 --test_hold 20 --lod_min 1 --lod_max 3 #--increase_lod_num_childs 2
# python train.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/downsample_3.5 --downsampling 3.5 --test_hold 20 --lod_min 1 --lod_max 2 --increase_lod_num_childs 2
# python train_lod.py -s /home/cglab/project/Dataset_opensource/MatrixCity_unzip/matrix_city_aerial/train/block_all -m results/MatrixCity/block_1/changing_LoD_v1/downsample_3.5/ --downsampling 3.5 --test_hold 20 --lod_min 1 --lod_max 3 --increase_lod_num_childs 2

# Previous command preserved for reference
# python train_lod.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/test/downsample_3.5/ --downsampling 3.5 --test_hold 20 --lod_min 1 --lod_max 3 --increase_lod_num_childs 2

# SG Training Command
# Uses --sh_degree 3 to ensure standard SH rasterization compatibility (SGs are projected to SH deg 3)
python train_lod.py \
    -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden \
    -m results/Mip-NeRF360/garden/sg_optimization/downsample_3.5/ \
    --downsampling 3.5 \
    --test_hold 20 \
    --lod_min 1 \
    --lod_max 3 \
    --increase_lod_num_childs 2 \
    --sh_degree 3