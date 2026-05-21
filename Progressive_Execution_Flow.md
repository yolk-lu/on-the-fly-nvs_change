# Progressive Pipeline Execution Workflow

本文記錄目前我們的 progressive anchor-local pipeline 從訓練指令啟動後，實際經歷的方法、函式與資料流。一般流程描述使用中文；類別、函式、參數、資料結構名稱維持英文。

## 0. Entrypoint

目前完整訓練腳本是：

```bash
./scripts/run_plan_training_full.sh
```

此腳本預設設定：

- `TRAIN_PY="${PROJECT_ROOT}/Progressive_train.py"`
- `PYTHON_BIN="/home/cglab/miniconda/envs/onthefly_nvs/bin/python"`
- `--progressive_overlap_mode reserved`
- `--progressive_anchor_radius`
- `--progressive_anchor_min_keyframes`
- `--progressive_loop_max_candidates`
- `--progressive_loop_min_anchor_gap`

也就是說，目前我們的新 pipeline 不再從舊 `train.py` 或 `train_lod.py` 執行。若直接執行舊 `train.py`，會走原本舊架構，不是本文記錄的 progressive pipeline。

本階段不使用 `viewer`、`VGGT pose prior`、`LSF velocity gate`。這三者不再由 `run_plan_training_full.sh` 傳入，也不屬於 progressive 專屬 config 的可調主參數。

執行順序：

```text
run_plan_training_full.sh
  -> python Progressive_train.py
      -> main()
      -> _parse_progressive_args(...)
      -> ProgressiveTrainer(args, cfg).run()
```

## 1. Argument / Manifest Flow

`Progressive_train.py` 啟動後直接解析 `pipeline.progressive_config.ProgressiveConfig`。目前 progressive pipeline 不再呼叫舊的 `args.get_args()`，也不再把完整 `args.py` namespace 傳入新流程。

主要函式：

- `_parse_progressive_args(argv)`
  - 呼叫 `parse_progressive_config(argv)`
  - 回傳 `ProgressiveConfig`
  - `ProgressiveConfig` 內部分成 `DataConfig`、`FeatureConfig`、`PoseConfig`、`AnchorConfig`、`LossConfig`、`OptimizerConfig`、`SpawnConfig`、`OutputConfig`、`ScaleConfig`
  - 解析 `--progressive_overlap_mode`
  - 解析 `--progressive_anchor_radius`
  - 解析 `--progressive_anchor_min_keyframes`
  - 解析 `--progressive_async_mapping`
  - 解析 `--progressive_tsdf_loss_weight`
  - 解析 `--progressive_anisotropy_loss_weight`
  - 解析 `--progressive_local_spawn_max`
  - 解析 `--progressive_local_spawn_target`
  - 解析 `--progressive_surface_sample_floor`
  - 解析 `--progressive_low_frequency_spawn_fraction`
  - 解析 `--progressive_edge_probability_threshold`
  - 解析 `--progressive_spawn_opacity_init`
  - 解析 `--progressive_anchor_render_check_every`
  - 解析 `--progressive_anchor_iterations`
  - 解析 `--progressive_loop_max_candidates`
  - 解析 `--progressive_loop_min_anchor_gap`
  - 固定 `use_vggt_pose_prior=False`
  - 固定 `pose_use_lsf_velocity_gate=False`

- `_write_manifest(...)`
  - 訓練完成或失敗後寫出 `progressive_manifest.json`
  - 會記錄 `progressive_config`
  - 會記錄 `pose_graph_optimization: sim3_enabled_after_verified_loop`
  - 會檢查 `metadata.json`、`point_clouds/`、`anchor_states/`、`tsdf/`、`colmap/` 等輸出是否完整

若訓練過程發生 exception：

```text
main()
  -> except Exception
  -> _write_manifest(status="failed", error=...)
  -> raise
```

## 2. ProgressiveTrainer Initialization

`ProgressiveTrainer.__init__(args, cfg)` 只建立狀態欄位，不載入 heavy module。

主要狀態：

- `self.dataset`
- `self.matcher`
- `self.triangulator`
- `self.pose_initializer`
- `self.dense_extractor`
- `self.depth_estimator`
- `self.observation_builder`
- `self.keyframe_store`
- `self.controller`
- `self.anchor_scene_model`
- `self.loop_manager`
- `self.tsdf_fusion`
- `self.spawn_policy`
- `self.pending_pose_queue`
- `self.bootstrap_frames`
- `self.bootstrap_desc_kpts`
- `self.frame_states`
- `self.loss_records`
- `self.current_lod`

接著 `ProgressiveTrainer.run()` 會先呼叫：

```python
self.initialize()
```

## 3. initialize()

`ProgressiveTrainer.initialize()` 是整個 pipeline 的模組建構階段。

### 3.1 Dataset

若 `args.source_path` 包含 `://`：

```python
self.dataset = StreamDataset(...)
```

否則：

```python
self.dataset = ImageDataset(self.args)
```

接著取得影像尺寸：

```python
self.height, self.width = self.dataset.get_image_size()
```

並設定：

- `self.max_error`
- `self.min_displacement`

### 3.2 Feature / Matching / Pose

建立 feature matcher：

```python
self.matcher = Matcher(...)
```

建立 triangulation module：

```python
self.triangulator = Triangulator(...)
```

建立 pose initializer：

```python
self.pose_initializer = PoseInitializer(
    self.width,
    self.height,
    self.triangulator,
    self.matcher,
    2 * self.max_error,
    self.args,
)
```

`PoseInitializer` 目前負責：

- bootstrap pose initialization
- incremental pose initialization
- `PnPRANSAC`
- `MiniBA`
- optional `ParallaxBA`
- failure log

### 3.3 Depth / Dense / Observation

建立 image feature / depth modules：

```python
self.dense_extractor = DenseExtractor(self.width, self.height)
self.depth_estimator = MonoDepthEstimator(self.width, self.height)
self.detector = Detector(...)
```

建立 observation builder：

```python
self.observation_builder = ObservationBuilder(
    detector=self.detector,
    depth_estimator=self.depth_estimator,
    dense_extractor=self.dense_extractor,
    matcher=self.matcher,
)
```

`ObservationBuilder.build(...)` 會把 raw image 轉成 `FrameState`：

- `image`
- `info`
- `desc_kpts`
- `dense_features`
- `mono_idepth`
- `mono_depth_conf`
- `frame_id`
- `mask`

`ObservationBuilder.match(left, right, **kwargs)` 會呼叫 `Matcher`，回傳 `Matches`。

### 3.4 Keyframe Store

建立我們自己的 keyframe store：

```python
self.keyframe_store = TrackingKeyframeStore(
    self.width,
    self.height,
    self.args,
    self.matcher,
    self.triangulator,
    device="cuda",
)
```

這個 store 取代舊 `SceneModel.keyframes`。

核心類別：

- `TrackingKeyframe`
- `TrackingKeyframeStore`

`TrackingKeyframe` 提供 pose initializer 和 renderer 需要的 keyframe interface：

- `get_R()`
- `get_t()`
- `get_Rt()`
- `set_Rt(Rt)`
- `get_centre(approx=False)`
- `get_mono_idepth(lvl=0)`
- `sample_conf(uv)`
- `update_3dpts(all_keyframes)`
- `to_json()`
- `to_colmap(id)`

`TrackingKeyframeStore` 負責：

- `add(frame, Rt, focal, index)`
- `add_keyframe(...)`
- `by_index(index)`
- `by_frame_id(frame_id)`
- `recent(n)`
- `get_prev_keyframes(n, update_3dpts, desc_kpts=None)`
- `refresh_after_pose_updates()`

### 3.5 Reconstruction Controller

建立 anchor / mapping / graph controller：

```python
self.controller = ReconstructionController(
    sh_degree=self.args.sh_degree,
    max_active_anchors=3,
    device="cuda",
    mapping_callback=self._mapping_step,
)
self.controller.create_anchor()
```

`ReconstructionController` 管理：

- `AnchorLocalMap`
- `ReadWriteLock`
- `AnchorGraph`
- `AnchorChunkManager`
- `MappingTask`
- mapping worker queue
- inter-anchor scale alignment
- loop edge verification

目前預設是同步 mapping；若設定 `--progressive_async_mapping`，會啟動：

```python
self.controller.start_mapping_worker()
```

### 3.6 Loop Closure Manager

建立 loop closure manager：

```python
self.loop_manager = LoopClosureManager(
    min_anchor_gap=self.cfg.loop_min_anchor_gap,
    max_candidates=self.cfg.loop_check_max_candidates,
)
```

目前 `LoopClosureManager` 只做：

- loop candidate discovery
- representative keyframe sampling
- matcher callback verification
- `AnchorGraph.add_loop_edge(...)`
- verified loop edge 後觸發 `AnchorPoseGraphOptimizer`
- event summary

目前 PGO 是 `Sim(3)` 第一版，只更新 anchor global similarity pose，不重訓 local Gaussians。metadata 會記錄：

```text
pose_graph_optimization: sim3_enabled_after_verified_loop
```

### 3.7 Progressive Scene Model

建立 anchor-local render / loss / optimizer layer：

```python
self.anchor_scene_model = ProgressiveSceneModel(
    controller=self.controller,
    width=self.width,
    height=self.height,
    f=self.state.focal_px,
    sh_degree=self.args.sh_degree,
    lambda_dssim=self.args.lambda_dssim,
    depth_loss_weight=self.args.depth_loss_weight_init,
    tsdf_loss_weight=self.cfg.tsdf_loss_weight,
    anisotropy_loss_weight=self.cfg.anisotropy_loss_weight,
    max_gaussian_aspect_ratio=...,
    lr_by_name={...},
)
```

`ProgressiveSceneModel` 內部會建立：

```python
self.renderer = AnchorLocalRenderer(...)
```

並提供：

- `render_from_keyframe(...)`
- `render(...)`
- `loss_from_keyframe(...)`
- `optimization_step(...)`
- `optimization_loop(...)`
- `anchor_regularization_losses()`
- `save(...)`

## 4. Main Training Loop

`ProgressiveTrainer.run()` 執行主 loop：

```python
for frame_id in tqdm(range(0, len(self.dataset))):
    self._process_frame(frame_id, pbar)
```

目前 viewer 不屬於 progressive pipeline 執行流程。

## 5. Per-Frame Flow: _process_frame()

每個 frame 的入口是：

```python
ProgressiveTrainer._process_frame(frame_id, pbar)
```

流程：

1. 從 dataset 讀取：

```python
image, info = self.dataset.getnext()
```

2. 建立 `FrameState`：

```python
frame = self.observation_builder.build(image, info, int(frame_id))
```

3. 保存到 frame lookup：

```python
self.frame_states[int(frame.frame_id)] = frame
```

4. 如果是第一個 keyframe：

```python
self.bootstrap_frames = [frame]
self.bootstrap_desc_kpts = [desc_kpts]
self.prev_frame = frame
self.state.n_keyframes += 1
return
```

5. 不是第一個 frame 時，與 `self.prev_frame` 做 matching：

```python
curr_prev_matches = self.observation_builder.match(frame, self.prev_frame)
dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
```

6. 判斷是否要新增 keyframe：

```python
should_add_keyframe = (
    dist.median() > self.min_displacement
    and len(curr_prev_matches.kpts) > self.args.min_num_inliers
)
should_add_keyframe |= info["is_test"]
```

7. 若一般位移規則沒有選為 keyframe，會執行 transient loop closure keyframe promotion：

```python
should_add_keyframe = self._promote_loop_candidate_keyframe(frame)
```

`_promote_loop_candidate_keyframe(...)` 會用 `DINOv2GlobalDescriptorExtractor + VLADAggregator` 對目前 input frame 查詢歷史 `AnchorDescriptorIndex`。若 retrieval 通過 `AdaptiveSimilarityThreshold`，再用 local feature matching 與 `LoopClosureVerifier` 檢查 `homography inlier ratio`。通過時才把目前 frame 提升為 keyframe，並把 `progressive_loop_candidates` 暫存在 `frame.info`，等 keyframe 被 pose initialize 並 attach 到 active anchor 後再寫入 loop edge。

8. 若要新增 keyframe：

```python
extra_registered = self._register_keyframe_candidate(frame)
```

9. 若註冊成功：

```python
self.state.n_keyframes += 1 + extra_registered
self.prev_frame = frame
self._evaluate_and_checkpoint(frame_id)
self._update_progress_bar(pbar)
```

## 6. Bootstrap Flow

bootstrap 由 `_register_keyframe_candidate(frame)` 控制。

在 keyframe 數量尚未到達：

```python
self.args.num_keyframes_miniba_bootstrap
```

時，會累積：

- `self.bootstrap_keyframe_dicts`
- `self.bootstrap_frames`
- `self.bootstrap_desc_kpts`

當達到 bootstrap 門檻：

```python
self._bootstrap_scene()
```

### 6.1 _bootstrap_scene()

1. 使用 `PoseInitializer.initialize_bootstrap(...)`：

```python
Rts, f, _ = self.pose_initializer.initialize_bootstrap(self.bootstrap_desc_kpts)
```

2. 更新 focal：

```python
self.state.focal_px = float(f.detach().cpu().item())
self.keyframe_store.f = self.state.focal_px
self.anchor_scene_model.update_intrinsics(f)
```

3. 對每個 bootstrap frame 建立 `TrackingKeyframe`：

```python
keyframe = self._make_keyframe(frame, Rt, index, focal)
```

`_make_keyframe(...)` 內部呼叫：

```python
self.keyframe_store.add_keyframe(frame, Rt, focal, index)
```

4. 把 keyframe 掛到 controller / anchor：

```python
self._attach_frame_to_controller(frame, keyframe)
```

5. 對 bootstrap keyframes 執行 anchor-local optimization：

```python
self._optimize_anchor_scene(frame, keyframe, "anchor_bootstrap_opt")
```

## 7. Incremental Pose Flow

當 bootstrap 完成後，新 keyframe 走：

```python
_register_incremental_keyframe(image, info, desc_kpts)
```

### 7.1 Select Previous Keyframes

先從我們自己的 keyframe store 選前序 keyframes：

```python
prev_keyframes = self.keyframe_store.get_prev_keyframes(
    self.args.num_prev_keyframes_miniba_incr,
    True,
    desc_kpts,
)
```

`get_prev_keyframes(...)` 會：

- 使用 camera centre distance 排序
- 若提供 `desc_kpts`，會用 `matcher.evaluate_match(...)` 做 match-score ranking
- 若 `update_3dpts=True`，會呼叫每個前序 keyframe 的：

```python
TrackingKeyframe.update_3dpts(self.keyframes)
```

`update_3dpts(...)` 會：

- 使用 `latest_invdepth` 更新已知 3D points
- 使用 `Triangulator.prepare_matches(...)`
- 使用 `Triangulator(...)`
- 更新 `desc_kpts.pts3d`
- 更新 `desc_kpts.depth`
- 更新 `desc_kpts.pts_conf`

### 7.2 Pose Initialization

接著呼叫：

```python
Rt = self.pose_initializer.initialize_incremental(
    prev_keyframes,
    desc_kpts,
    self.state.n_keyframes,
    info["is_test"],
    image,
    all_keyframes=self.keyframe_store.keyframes,
    retry_count=0,
    frame_uid=info.get("frame_id"),
)
```

`PoseInitializer.initialize_incremental(...)` 主要使用：

- current frame `desc_kpts`
- previous `TrackingKeyframe.desc_kpts`
- existing `desc_kpts.pts3d`
- `PnPRANSAC`
- `MiniBA`
- optional velocity consistency gate

若 `Rt is None`：

```python
self._queue_failed_pose_candidate(...)
return -1
```

若成功：

```python
self._add_initialized_keyframe(...)
return self._retry_pending_keyframes()
```

## 8. Add Keyframe Flow

`_add_initialized_keyframe(...)` 負責把 incremental pose 成功的 frame 放進 pipeline。

流程：

1. 若使用 COLMAP pose：

```python
if self.args.use_colmap_poses:
    Rt = info["Rt"]
```

2. 建立 `TrackingKeyframe`：

```python
keyframe = self._make_keyframe(frame, Rt, index, focal)
```

3. 掛到 controller：

```python
self._attach_frame_to_controller(frame, keyframe)
```

4. 執行 active anchor optimization：

```python
self._optimize_anchor_scene(frame, keyframe, f"anchor_{phase}")
```

5. 記錄 loss：

```python
_append_loss_record(...)
```

## 9. Anchor Attachment / Rollover Flow

所有新 keyframe 都會經過：

```python
_attach_frame_to_controller(frame, keyframe)
```

### 9.1 First Anchor Pose

如果 active anchor 尚未有 keyframe：

```python
active_anchor.t_anchor_to_world = cam_centre
self.controller.graph.add_node(...)
```

### 9.2 Anchor Rollover / Seal Rule

`Anchor` 的主要目的不是單純用空間半徑切段，而是作為 active local Gaussian scene chunk。也就是：

- `add new gaussians` 只寫入目前 active anchor
- `optimize` 只針對目前 active anchor 或 active set
- `merge gaussians` 在 active anchor 封存前執行
- active anchor 達到容量上限時，先儲存已最佳化的 local scene，再建立下一個 active anchor

若 active anchor 已有 keyframes，會先計算 budget status：

```python
self.controller.anchor_budget_status(
    cam_centre,
    max_anchor_radius=self.cfg.anchor_radius,
    min_keyframes=self.cfg.anchor_min_keyframes,
    max_gaussians=self.cfg.max_anchor_gaussians,
    max_keyframes=self.cfg.max_anchor_keyframes,
    max_tsdf_voxels=self.cfg.max_anchor_tsdf_voxels,
    max_vram_mb=self.cfg.max_anchor_vram_mb,
)
```

目前任一條件成立就會 seal active anchor：

- `num_gaussians >= max_anchor_gaussians`
- `num_keyframes >= max_anchor_keyframes`
- `num_tsdf_voxels >= max_anchor_tsdf_voxels`
- `gpu_used_mb >= max_anchor_vram_mb`，若 `max_anchor_vram_mb > 0`
- `distance(camera_center, anchor_origin) > anchor_radius` 且 `num_keyframes >= anchor_min_keyframes`

seal 流程：

```text
_seal_anchor_and_start_next(...)
  -> AnchorFinalOpt
  -> ProgressiveSceneModel.merge_anchor_gaussians(...)
  -> ProgressiveSceneModel.save_anchor(...)
  -> ReconstructionController.create_active_anchor(...)
```

當新 active anchor 建立：

```python
self.controller.create_active_anchor(
    torch.eye(3, device=cam_centre.device, dtype=cam_centre.dtype),
    cam_centre,
    reference_frame=self.prev_frame,
    new_frame=frame,
)
```

`ReconstructionController.create_active_anchor(...)` 會：

- `seal_active_anchor()`
- 建立新的 `AnchorLocalMap`
- 執行 `align_inter_anchor_scale(reference_frame, new_frame)`
- 寫入 `AnchorGraph` node
- 建立 sequential edge

### 9.2.1 Scale Formula

`AnchorLocalMap` 的 local-to-world 定義固定為：

```text
x_world = s_anchor_to_world * R_anchor_to_world * x_local + t_anchor_to_world
```

目前 inter-anchor monocular scale alignment 使用 inverse-depth ratio：

```text
cell_scale = reference_idepth / new_idepth
global_scale = median(valid cell_scale)
new_anchor.s_anchor_to_world = previous_anchor.s_anchor_to_world * global_scale
```

有效 cell 條件：

- `reference_idepth` 與 `new_idepth` finite
- `idepth > 1e-6`
- `reference_conf > min_conf`
- `new_conf > min_conf`
- cell 內樣本數 `>= min_samples_per_cell`

當 PGO 回寫 anchor pose 時，scale 使用 `Sim(3)` 的 `log_scale` 狀態：

```text
s_anchor_to_world = exp(log_scale)
```

Gaussian covariance 在 anchor scale 更新後使用 similarity transform 補償：

```text
cov_new = (s_old / s_new)^2 * R_delta.T @ cov_old @ R_delta
R_delta = R_old.T @ R_new
```

### 9.3 Register Frame

不論是否 rollover，最後都會：

```python
self.controller.add_frame(frame, active_anchor)
frame.info["progressive_anchor_id"] = active_anchor.anchor_id
frame.info["progressive_keyframe_index"] = int(keyframe.index)
```

如果剛發生 rollover，會執行：

```python
self._check_loop_closure(active_anchor.anchor_id)
```

### 9.4 Mapping Dispatch

同步模式：

```python
self._mapping_step(MappingTask(frame=frame, anchor_id=active_anchor.anchor_id))
```

非同步模式：

```python
self.controller.enqueue_mapping_task(frame, active_anchor.anchor_id)
```

非同步 worker 由 `ReconstructionController._mapping_loop()` 消耗 `MappingTask`，並呼叫：

```python
self.mapping_callback(task)
```

此 callback 在 trainer 內就是：

```python
ProgressiveTrainer._mapping_step(...)
```

## 10. Mapping Step

mapping 的核心函式：

```python
_mapping_step(task: MappingTask)
```

流程：

1. 取得 anchor：

```python
anchor = self.controller.anchors[int(task.anchor_id)]
```

2. 透過 frame metadata 找 keyframe：

```python
keyframe_index = int(frame.info.get("progressive_keyframe_index", -1))
keyframe = self.keyframe_store.by_index(keyframe_index)
```

3. 從 `keyframe.get_Rt()` 計算 camera-to-world：

```python
Rt = keyframe.get_Rt().detach()
R_w2c = Rt[:3, :3]
t_w2c = Rt[:3, 3]
R_cam_to_world = R_w2c.T
t_cam_to_world = -R_w2c.T @ t_w2c
```

4. 對 anchor 加 write lock：

```python
with self.controller.anchor_locks[anchor.anchor_id].write_lock():
    ...
```

5. 融合 depth 到 local TSDF：

```python
self.tsdf_fusion.integrate_depth(
    anchor.tsdf,
    frame.mono_idepth,
    frame.mono_depth_conf,
    anchor.R_world_to_anchor,
    anchor.t_world_to_anchor,
    R_cam_to_world,
    t_cam_to_world,
    self.keyframe_store.f,
    self.keyframe_store.centre,
)
```

`TSDFFusion.integrate_depth(...)` 會：

- 將 monocular inverse depth 轉成 depth samples
- 使用 confidence gate
- 建立 camera-space points
- 轉到 world space
- 再轉到 anchor-local space
- 呼叫 `AdaptiveTSDF.integrate(...)`

6. 生成 anchor-local Gaussians。若 active anchor 已有 Gaussian，會先從目前 keyframe render 一次，作為 spawn penalty / occlusion reference：

```python
result = self.anchor_scene_model.render_from_keyframe(keyframe, active_anchor_ids=[anchor.anchor_id])
self._spawn_anchor_local_gaussians(
    anchor,
    frame,
    R_cam_to_world,
    t_cam_to_world,
    rendered_image=result.render,
    rendered_invdepth=result.invdepth,
)
```

7. 做 periodic render diagnostics：

```python
self._maybe_anchor_render_check(frame, anchor)
```

## 11. Gaussian Spawn Flow

Gaussian 生成由：

```python
_spawn_anchor_local_gaussians(...)
```

負責。

第一次使用時建立：

```python
self.spawn_policy = GaussianSpawnPolicy(
    width=self.width,
    height=self.height,
    f=self.keyframe_store.f,
    centre=self.keyframe_store.centre,
    init_proba_scaler=self.args.init_proba_scaler,
    target_samples=self.cfg.local_spawn_target,
    surface_sample_floor=self.cfg.surface_sample_floor,
    low_frequency_fraction=self.cfg.low_frequency_spawn_fraction,
    edge_probability_threshold=self.cfg.edge_probability_threshold,
)
```

接著：

```python
spawn = self.spawn_policy.sample(
    frame,
    rendered_image=rendered_image,
    rendered_invdepth=rendered_invdepth,
)
```

`GaussianSpawnPolicy.sample(frame)` 目前負責：

- 依 `frame.mono_idepth`
- 依 `frame.mono_depth_conf`
- 依 image high-frequency / confidence sampling
- 使用 `edge_proba = clamp(LoG(input) - LoG(rendered), 0, 1)` 作為主要 sampling probability，高頻邊界會優先取得更多 Gaussian
- 使用 `surface_sample_floor * mono_depth_conf` 在平滑但有深度信心的區域保留最低 sampling probability
- 使用 `local_spawn_target` 做 quota-based sampling；補點時先補高頻區域，低頻區域最多只佔 `low_frequency_spawn_fraction`，避免平滑區被大量 Gaussian 填滿
- `edge_probability_threshold` 定義高頻與低頻區域的分界
- 若有 `rendered_image`，使用 rendered image 的 `Laplacian of Gaussian` 當 penalty，降低已被目前 Gaussian 解釋區域的重複 sampling
- 若有 `rendered_invdepth`，做 occlusion gate，只在 monocular depth 比 rendered depth 更靠近相機時新增點
- 輸出 `GaussianSpawnResult`

`GaussianSpawnResult` 包含：

- `xyz_cam`
- `f_dc`
- `depth`
- `init_probability`
- `source`
- `stats`

trainer 會把 `xyz_cam` 轉成 world，再轉成 anchor-local：

```python
xyz_world = (R_cam_to_world @ xyz_cam.T).T + t_cam_to_world[None]
xyz_local = anchor.world_to_local(xyz_world)
```

然後建立 local Gaussian extension：

```python
extension = {
    "xyz": xyz_local,
    "f_dc": f_dc,
    "f_rest": zeros,
    "opacity": inverse_sigmoid(progressive_spawn_opacity_init),
    "scaling": log(depth / (focal * sqrt(init_probability))),
    "rotation": identity_quaternion,
}
```

最後做 finite gate：

```python
finite &= torch.isfinite(tensor.flatten(1)).all(dim=1)
```

通過後：

```python
anchor.gaussian_model.append(extension, anchor.anchor_id)
```

每次 spawn 後會寫入 `last_spawn` diagnostics，periodic `AnchorRender` 時會同步輸出：

- `spawned`：depth gate 後的 Gaussian 候選數
- `kept_after_cap`：套用 `local_spawn_max` 後實際保留數
- `finite_after_transform`：轉成 world / anchor-local 後仍 finite 的數量
- `reprojection_error_p95_px`：`uv -> xyz_cam -> world -> camera -> uv` 的 95% re-projection error，用來確認點是否真的放在輸入影像射線上
- `tsdf_valid_ratio`：新增點落在已觀測 TSDF voxel 的比例，用來確認是否貼近 TSDF physical surface
- `camera_depth_median`：新增點回到目前 camera frame 後的 depth median，用來檢查 monocular depth scale 是否爆掉

## 12. Anchor Render / Loss / Optimization

active anchor optimization 入口：

```python
_optimize_anchor_scene(frame, keyframe, phase)
```

若 `self.cfg.anchor_iterations <= 0` 或 `self.cfg.async_mapping=True`，此函式會跳過。

否則：

```python
self.anchor_scene_model.optimization_loop(
    keyframe,
    frame,
    anchor_id=anchor_id,
    n_iters=self.cfg.anchor_iterations,
)
```

### 12.1 ProgressiveSceneModel.optimization_loop()

`optimization_loop(...)` 會重複呼叫：

```python
optimization_step_multiview(anchor_id, view_items)
```

`view_items` 由 `ProgressiveTrainer._select_anchor_training_views(...)` 建立：

- 第一個 view 固定是目前新增或目前正在處理的 `TrackingKeyframe`
- 後續 views 從同一個 active anchor 的 `anchor.keyframe_ids` 中反向挑選近期 keyframes
- 數量由 `progressive_anchor_train_views` 控制，預設為 `4`

因此 optimize 不再只看單張圖，而是對 active anchor 的多視角重建誤差取平均。

### 12.2 ProgressiveSceneModel.optimization_step_multiview()

流程：

1. 取得 anchor optimizer：

```python
optimizer = self._optimizer_for_anchor(anchor)
```

2. 對每個 selected view 計算 losses：

```python
per_view_losses = [
    self.loss_from_keyframe(kf, frame, active_anchor_ids=[anchor.anchor_id])
    for kf, frame in view_items
]
total_loss = torch.stack([loss.total for loss in per_view_losses]).mean()
```

3. 若 loss finite 且有 grad：

```python
total_loss.backward()
torch.nn.utils.clip_grad_norm_(...)
optimizer.step()
self._sanitize_anchor_params(anchor)
```

4. 記錄 optimizer diagnostics：

```python
grad_xyz_mean / grad_xyz_max
grad_opacity_mean / grad_opacity_max
grad_scaling_mean / grad_scaling_max
depth_valid_pixels
visible_gaussians
num_views
```

### 12.3 ProgressiveSceneModel.loss_from_keyframe()

`loss_from_keyframe(...)` 會：

1. 呼叫：

```python
result = self.render_from_keyframe(keyframe, active_anchor_ids=...)
```

2. 計算 RGB loss：

```python
rgb_loss = (result_image - gt_image).abs().mean()
```

3. 計算 SSIM loss：

```python
ssim_loss = 1 - fused_ssim(...)
```

4. 計算 monocular depth loss，但只在 renderer 真的產生有效 `invdepth` 的 pixel 上計算：

```python
valid_depth =
    isfinite(invdepth)
    & isfinite(mono_idepth)
    & (invdepth > progressive_depth_valid_epsilon)
    & (mono_idepth > progressive_depth_valid_epsilon)
depth_loss = abs(invdepth[valid_depth] - mono_idepth[valid_depth]).mean()
```

這個改動避免未渲染背景 `invdepth = 0` 被拿去和 positive monocular inverse depth 做全圖懲罰，否則 loss 會把 Gaussian 往錯誤方向推。

5. 計算 anchor regularization：

```python
tsdf_loss, anisotropy_loss = self.anchor_regularization_losses()
```

6. 合成 total loss：

```python
total =
    lambda_dssim * ssim_loss
    + (1 - lambda_dssim) * rgb_loss
    + depth_loss_weight * depth_loss
    + tsdf_loss_weight * tsdf_loss
    + anisotropy_loss_weight * anisotropy_loss
```

## 13. AnchorLocalRenderer Flow

render 入口：

```python
ProgressiveSceneModel.render_from_keyframe(keyframe, active_anchor_ids)
```

內部轉成：

```python
view_matrix = keyframe.get_Rt().transpose(0, 1)
self.render(view_matrix, keyframe.get_centre(approx=True), active_anchor_ids)
```

`ProgressiveSceneModel.render(...)` 會：

1. 若未指定 active anchors：

```python
active_anchor_ids = self.controller.update_active_set(cam_centre_world)
```

2. 收集 anchors：

```python
anchors = [self.controller.anchors[int(anchor_id)] for anchor_id in active_anchor_ids]
```

3. 呼叫：

```python
self.renderer.render(anchors, view_matrix)
```

### 13.1 AnchorLocalRenderer.collect_anchor_batch()

`collect_anchor_batch(...)` 會把多個 `AnchorLocalMap` 的 Gaussian 參數轉成 world-space batch。

每個 Gaussian 使用固定欄位：

```python
GAUSSIAN_KEYS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
```

流程：

- 呼叫 `anchor.gaussian_model.world_params(...)`
- 保留 `anchor_ids`
- concat all active anchors
- 回傳 `AnchorRenderBatch`

### 13.2 AnchorLocalRenderer.render()

render 前會先執行：

```python
RenderGuard.filter(...)
```

`RenderGuard` 負責：

- non-finite check
- distance / depth check
- screen-size check

通過後才呼叫 Gaussian rasterizer。

render 結果包成 `AnchorRenderResult`：

- `render`
- `invdepth`
- `radii`
- `visibility_filter`
- `anchor_ids`
- `guard_mask`
- `cuda_error`

`ProgressiveSceneModel._render_debug(...)` 會記錄：

- `num_input_gaussians`
- `num_visible_gaussians`
- `num_positive_radii`
- `visible_anchor_ids`
- `guard_reason_counts`
- `cuda_error`

## 14. Loop Closure Flow

loop closure 有兩個入口：

1. `anchor rollover` 後檢查新 anchor 是否和歷史 anchor 閉環。
2. 一般 frame 沒通過 keyframe 位移規則時，先用 DINOv2/VLAD 檢索歷史相似 keyframe；若 local feature/homography 通過，將此 frame 提升為 keyframe，再建立 frame-level loop edge。

anchor rollover 後檢查：

```python
_check_loop_closure(new_anchor_id)
```

此函式建立 matcher callback：

```python
def matcher_fn(left, right):
    return self.observation_builder.match(left, right, remove_outliers=False)
```

然後呼叫：

```python
self.loop_manager.check_anchor_rollover(
    new_anchor_id,
    self.keyframe_store,
    self.controller,
    matcher_fn,
)
```

### 14.1 LoopClosureManager.check_anchor_rollover()

流程：

1. 若 anchor 少於 3 個，直接跳過。
2. 呼叫 DINOv2/VLAD retrieval，並用 bbox/proximity fallback 補候選。
3. 對每個 candidate：
   - 取 source representative keyframe
   - 取 destination representative keyframe
   - 計算 `T_src_to_dst`
   - 呼叫 `controller.verify_and_add_loop_edge(...)`

### 14.2 Frame-Level Loop Keyframe Promotion

當 `_process_frame()` 的位移規則不選 keyframe 時，執行：

```python
candidates = self.place_index.query_frame(
    frame,
    current_anchor_id=active_anchor_id,
    exclude_anchor_window=self.cfg.loop_min_anchor_gap,
    top_k=self.cfg.loop_check_max_candidates,
)
```

對每個 retrieval candidate：

- 取 `candidate.keyframe_index` 對應的 historical `TrackingKeyframe`
- 使用 `LoopClosureVerifier.verify(frame, dst_keyframe.frame, matcher_fn)`
- 若通過，將 candidate 存入 `frame.info["progressive_loop_candidates"]`
- 此 frame 會被提升為 keyframe，進入正常 pose initialize / attach / mapping / optimize 流程

attach 到 active anchor 後呼叫：

```python
self.loop_manager.check_frame_candidates(...)
```

通過 verification 時才呼叫：

```python
controller.verify_and_add_loop_edge(...)
controller.optimize_anchor_graph(...)
```

### 14.3 Candidate Discovery

Candidate discovery 會：

- 排除自己
- 排除 anchor id 距離小於 `min_anchor_gap` 的相鄰 anchors
- 優先使用 `AnchorDescriptorIndex.query_anchor(...)` 或 `query_frame(...)`
- 使用 anchor bbox overlap
- 使用 anchor centre proximity
- 最多回傳 `max_candidates`

### 14.4 Geometric Verification

`ReconstructionController.verify_and_add_loop_edge(...)` 會呼叫：

```python
self.loop_verifier.verify(left_frame, right_frame, matcher_fn)
```

`LoopClosureVerifier.verify(...)` 會：

- 取得 `Matches.kpts`
- 取得 `Matches.kpts_other`
- 檢查 `min_matches`
- 使用 `cv2.findHomography(..., cv2.RANSAC, ransac_px)`
- 檢查 `min_inlier_ratio`

若通過：

```python
self.graph.add_loop_edge(src_anchor_id, dst_anchor_id, T_src_to_dst, weight)
```

目前不執行全域 `Pose Graph Optimization`，也不回寫 anchor pose。這是下一階段要補的 solver。

## 15. Pose Retry Flow

若 incremental pose initialization 失敗，會進入：

```python
_queue_failed_pose_candidate(...)
```

目前只有當：

```python
self.pose_initializer.last_failure_reason == "lsf_velocity_gate"
```

且 retry 參數允許時，才會放入：

```python
self.pending_pose_queue
```

當後續 keyframe 成功後：

```python
_retry_pending_keyframes()
```

會重試 pending frames。重試仍然走：

- `keyframe_store.get_prev_keyframes(...)`
- `pose_initializer.initialize_incremental(...)`
- `_add_initialized_keyframe(...)`

## 16. Reboot Flow

`_maybe_reboot()` 用於長序列 pose 退化時重跑一小段 bootstrap。

觸發條件：

- `args.enable_reboot=True`
- 已有 `keyframe_store.approx_cam_centres`
- keyframe 數量足夠
- 最近 camera centre motion 過大或過小
- 距離上次 reboot 超過門檻

觸發後：

```python
bs_kfs = self.keyframe_store.recent(8)
Rts, _, final_residual = self.pose_initializer.initialize_bootstrap(...)
```

若 residual 合格：

```python
Rts = align_mean_up_fwd(Rts, in_Rts)
keyframe.set_Rt(Rt)
self.keyframe_store.refresh_after_pose_updates()
```

目前 reboot 只更新 `TrackingKeyframe` pose，不重新建立舊 global scene。

## 17. Checkpoint / Save Flow

中途 checkpoint：

```python
_evaluate_and_checkpoint(frame_id)
```

若符合 `args.save_every`：

```python
self.anchor_scene_model.save(
    os.path.join(self.args.model_path, "progress", f"{frame_id:05d}"),
    self.keyframe_store.keyframes,
    n_frames=len(self.dataset),
)
```

訓練結束：

```python
_finalize(reconstruction_time)
```

流程：

1. 若 async mapping 啟用，等待 queue 完成：

```python
self.controller.mapping_queue.join()
self.controller.stop_mapping_worker()
```

2. 儲存 anchor-local scene：

```python
metrics = self.anchor_scene_model.save(
    self.args.model_path,
    self.keyframe_store.keyframes,
    reconstruction_time,
    len(self.dataset),
)
```

3. 儲存 pose failure log：

```python
self.pose_initializer.save_failure_log(self.args.model_path)
```

4. 儲存 runtime stats：

```python
self.tracker.save_stats(...)
```

5. 儲存 loss records：

```python
_save_loss_records_and_plot(...)
```

6. 儲存 LoD marker：

```python
_save_lod_completion_marker(...)
```

7. 回傳 summary：

```python
{
    "num_keyframes": ...,
    "num_anchors": ...,
    "pipeline_state": ...,
    "anchor_scene_state": ...,
    "loop_closure": ...,
    "num_loss_records": ...,
    "reconstruction_time": ...,
}
```

## 18. ProgressiveSceneModel.save()

`ProgressiveSceneModel.save(path, keyframes, reconstruction_time, n_frames)` 會寫出：

```text
metadata.json
point_clouds/anchor_*.ply
anchor_states/anchor_*.pt
tsdf/anchor_*.pt
colmap/cameras.bin
colmap/images.bin
```

### 18.1 Anchor State

每個 anchor 會保存：

- `anchor_id`
- `R_anchor_to_world`
- `t_anchor_to_world`
- `keyframe_ids`
- `gaussian_params`
- `anchor_ids`

### 18.2 TSDF State

每個 anchor 會保存：

- `base_voxel_size`
- `num_levels`
- `keys`
- `hashes`
- `tsdf_mean`
- `m2`
- `weight`
- `level`

### 18.3 Metadata

`metadata.json` 包含：

- metrics
- config
- anchors
- keyframes
- graph
- `pose_graph_optimization`

### 18.4 COLMAP Export

對每個 `TrackingKeyframe` 呼叫：

```python
keyframe.to_colmap(index)
```

產生：

- `Camera`
- `BaseImage`

最後呼叫：

```python
write_model(cameras, images, {}, colmap_save_path, ext=".bin")
```

## 19. Current Boundaries

目前已經切掉的舊依賴：

- `Progressive_train.py` 不再 import `scene.scene_model.SceneModel`
- `Progressive_train.py` 不再 import `scene.keyframe.Keyframe`
- `Progressive_train.py` 不再使用 `self.scene_model`
- checkpoint 與 final save 不再使用舊 `SceneModel.save(...)`
- Gaussian spawn / render / optimization 不再走舊 `SceneModel`

目前保留但尚未實作完整功能：

- `overlap optimization` 仍是 reserved
- `Pose Graph Optimization` 已有 `Sim(3)` 第一版，但只有 verified loop edge 後才會觸發
- viewer 不納入目前 progressive pipeline
- full global evaluation path 尚未接回 progressive pipeline

## 20. Compact Call Graph

```text
run_plan_training_full.sh
  -> Progressive_train.py
    -> main()
      -> _parse_progressive_args()
      -> ProgressiveTrainer(args, cfg).run()
        -> initialize()
          -> ImageDataset / StreamDataset
          -> Matcher
          -> Triangulator
          -> PoseInitializer
          -> DenseExtractor
          -> MonoDepthEstimator
          -> Detector
          -> ObservationBuilder
          -> TrackingKeyframeStore
          -> ReconstructionController.create_anchor()
          -> LoopClosureManager
          -> ProgressiveSceneModel
            -> AnchorLocalRenderer
        -> for frame_id in dataset:
          -> _process_frame()
            -> ObservationBuilder.build()
            -> ObservationBuilder.match()
            -> _register_keyframe_candidate()
              -> _bootstrap_scene()
                -> PoseInitializer.initialize_bootstrap()
                -> _make_keyframe()
                  -> TrackingKeyframeStore.add_keyframe()
                -> _attach_frame_to_controller()
                -> _mapping_step()
                -> _optimize_anchor_scene()
              -> _maybe_reboot()
              -> _register_incremental_keyframe()
                -> TrackingKeyframeStore.get_prev_keyframes()
                  -> TrackingKeyframe.update_3dpts()
                -> PoseInitializer.initialize_incremental()
                -> _add_initialized_keyframe()
                  -> _make_keyframe()
                  -> _attach_frame_to_controller()
                    -> ReconstructionController.should_roll_anchor()
                    -> ReconstructionController.create_active_anchor()
                    -> _check_loop_closure()
                      -> LoopClosureManager.check_anchor_rollover()
                      -> ReconstructionController.verify_and_add_loop_edge()
                    -> _mapping_step()
                      -> TSDFFusion.integrate_depth()
                      -> _spawn_anchor_local_gaussians()
                        -> GaussianSpawnPolicy.sample()
                      -> _maybe_anchor_render_check()
                  -> _optimize_anchor_scene()
                    -> ProgressiveSceneModel.optimization_loop()
                      -> optimization_step()
                        -> loss_from_keyframe()
                          -> render_from_keyframe()
                            -> AnchorLocalRenderer.render()
                              -> collect_anchor_batch()
                              -> RenderGuard.filter()
                              -> rasterizer
                        -> backward()
                        -> optimizer.step()
            -> _evaluate_and_checkpoint()
        -> _finalize()
          -> ProgressiveSceneModel.save()
          -> PoseInitializer.save_failure_log()
          -> ResourceTracker.save_stats()
          -> _save_loss_records_and_plot()
          -> _save_lod_completion_marker()
      -> _write_manifest()
```
