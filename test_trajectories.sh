python /home/cglab/project/on-the-fly-nvs/test_pose_trajectory.py -s /home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1_small \
  --downsampling 2 \
  --use_semantic_features \
  --semantic_backbone dinov2 \
  --sem_feat_dim 128 \
  --sem_weight 0.3 \
  --no_pose_use_lsf_velocity_gate
