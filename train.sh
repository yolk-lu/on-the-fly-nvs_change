

# python train.py -s /home/cglab/project/Dataset_opensource/Drone-dataset/chiayi_0326 -m results/Drone/Chiayi_0326 


# python train.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/downsample_4 --downsampling 4 --test_hold 20 --lod_min 1 --lod_max 3 #--increase_lod_num_childs 2
# python train.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/downsample_3.5 --downsampling 3.5 --test_hold 20 --lod_min 1 --lod_max 2 --increase_lod_num_childs 2
python train_lod.py -s /home/cglab/project/Dataset_opensource/Mip-NeRF360/garden -m results/Mip-NeRF360/garden/changing_LoD_v1/downsample_3.5 --downsampling 3.5 --test_hold 20 --lod_min 1 --lod_max 3 --increase_lod_num_childs 2