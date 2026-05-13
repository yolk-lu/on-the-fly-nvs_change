# train.py / train_lod.py Training Logic Review

本文整理既有 `train.py` 與 `train_lod.py` 的訓練流程，作為後續實作 `Progressive_train.py` 的依據。重點不是複製舊流程，而是確認舊流程中哪些責任需要被 anchor-local / Progressive 架構替換。

## 1. Shared Initialization Flow

兩個入口的前處理大致相同。

```text
parse args
  -> seed torch / cuda / numpy
  -> build ImageDataset or StreamDataset
  -> get image size
  -> build Matcher
  -> build Triangulator
  -> build PoseInitializer
  -> build DenseExtractor
  -> build MonoDepthEstimator
  -> build SceneModel
  -> optional SemanticExtractor
  -> build Detector
  -> optional viewer
```

主要模組責任：

| Module | Current responsibility |
| --- | --- |
| `Detector` | 對每張影像抽 keypoints / descriptors |
| `Matcher` | 相鄰 frame 或 keyframe 間特徵匹配 |
| `Triangulator` | keyframe 之間三角化 3D points，現在可接 ParallaxBA refinement |
| `PoseInitializer` | bootstrap / incremental pose 初始化 |
| `Keyframe` | 包裝 image pyramid、feature map、mono depth、pose/depth/exposure optimizer |
| `SceneModel` | 同時管理 keyframes、global gaussian params、render、optimization、anchors、save/eval |

Progressive 需要拆掉的核心耦合：

- 舊 `SceneModel` 是單一大型狀態容器。
- Progressive 不應建立單一全域 Gaussian cloud。
- Progressive 應把 tracking、mapping、local anchor、global graph 分開。

## 2. train.py Main Reconstruction Loop

`train.py` 是原始 on-the-fly 訓練主線，流程是同步式為主，stream 模式才用 `scene_model.optimize_async()`。

### 2.1 First Frame

```text
if n_keyframes == 0:
  image, info = dataset.getnext()
  prev_desc_kpts = detector(image)
  bootstrap_keyframe_dicts = [image/info]
  bootstrap_desc_kpts = [prev_desc_kpts]
  n_keyframes += 1
  continue
```

第一張只作為 bootstrap 起點，不建立 `Keyframe`，不 spawn Gaussians。

### 2.2 Keyframe Decision

每個新 frame：

```text
image, info = dataset.getnext()
desc_kpts = detector(image)
curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
dist = norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other)
should_add_keyframe =
  dist.median() > min_displacement
  and num_matches > min_num_inliers
should_add_keyframe |= info["is_test"]
```

舊流程假設 tracking 和 mapping 在同一 loop 中進行。Progressive 需要把「是否接受 keyframe」留在 tracking frontend，但把 mapping 工作 enqueue 到 backend。

### 2.3 Bootstrap

當累積到 `num_keyframes_miniba_bootstrap`：

```text
Rts, f, residual = pose_initializer.initialize_bootstrap(bootstrap_desc_kpts)
for each bootstrap frame:
  if use_colmap_poses: override Rt / f
  keyframe = Keyframe(...)
  scene_model.add_keyframe(keyframe, f)

for each bootstrap keyframe:
  scene_model.add_new_gaussians(index)

if stream:
  scene_model.optimize_async(num_iterations)
else:
  scene_model.optimization_loop(num_iterations)
```

Bootstrap 同時完成：

- 初始 pose graph / focal
- `Keyframe` 建立
- Gaussian direct sampling
- 初始 optimization

Progressive 對應：

- bootstrap pose 還是由 ParallaxBA / PoseInitializer 主導
- `Keyframe` 應拆成 `FrameState` + local train state
- Gaussian spawn 應寫入 active `AnchorLocalMap`
- optimization 應由 mapping thread 執行，不阻塞 tracking loop

### 2.4 Reboot

`train.py` 在 pose baseline 過大或過小時觸發 reboot：

```text
if enable_reboot and approx_cam_centres exists:
  rel_dist = mean distance over last 20 camera centers
  needs_reboot = rel_dist too large/small and enough frames since last reboot

if needs_reboot:
  bs_kfs = last 8 keyframes
  Rts, residual = initialize_bootstrap(bs_kfs, rebooting=True)
  if residual good:
    align Rts with current poses
    update keyframe poses
    scene_model.reset()
    re-add gaussians for last frames
    run 3 * num_iterations optimization_step()
```

Progressive 不應直接 reset 全域 scene。對應策略應改成：

- 對 active anchor 做 local recovery。
- 必要時封存當前 anchor，建立新 anchor。
- 不做全域 Gaussian reset。
- failure reason 需要寫入 tracking/mapping log。

### 2.5 Incremental Reconstruction

Bootstrap 完成後，每個 keyframe：

```text
prev_keyframes = scene_model.get_prev_keyframes(
  num_prev_keyframes_miniba_incr,
  update_3dpts=True,
  desc_kpts=desc_kpts,
)
Rt = pose_initializer.initialize_incremental(prev_keyframes, desc_kpts, ...)

if Rt is not None:
  keyframe = Keyframe(...)
  scene_model.add_keyframe(keyframe)
  scene_model.add_new_gaussians()
  scene_model.optimization_loop(num_iterations) or optimize_async()
else:
  should_add_keyframe = False
```

這是 Progressive 最需要拆分的地方：

- `get_prev_keyframes()` 目前會依賴 `SceneModel` 的 keyframe list、sorted indices、triangulation update。
- `initialize_incremental()` 目前需要舊 `Keyframe` API。
- `add_new_gaussians()` 同時做 depth alignment、guided MVS、direct sampling、occlusion pruning、optimizer add/prune。
- `optimization_loop()` 同步 block main loop。

Progressive 對應：

```text
Tracking frontend:
  select local/nearby frame states
  run ParallaxBA incremental pose
  decide keyframe
  enqueue MappingTask

Mapping backend:
  update local 3D observations
  spawn Gaussians into active anchor
  fuse TSDF
  optimize active window
```

### 2.6 Anchor Placement

`train.py` 每次接受 keyframe 後呼叫：

```text
scene_model.place_anchor_if_needed()
```

舊 `place_anchor_if_needed()` 會：

- 根據 Gaussian screen size 判斷 small Gaussian 比例。
- merge small Gaussians。
- 把舊 anchor offload 到 CPU。
- 建立新 active anchor。

Progressive 對應：

- anchor rollover 由 `ReconstructionController.should_roll_anchor()` 判斷。
- 封存由 `seal_active_anchor()` 執行。
- 建立新 anchor 時必須做 inter-anchor scale alignment。
- 不做不透明全域 merge；必須保持 LocalGaussianModel / AnchorLocalMap 邊界。

### 2.7 Save / Evaluation / Finetune

主 loop 結束：

```text
scene_model.enable_inference_mode()
metrics = scene_model.save(model_path, reconstruction_time, len(dataset))
pose_initializer.save_failure_log(model_path)
```

若 `save_at_finetune_epoch` 有設定：

```text
for epoch in finetune_epochs:
  scene_model.finetune_epoch()
  if epoch in save_at_finetune_epoch:
    scene_model.save(...)
```

Progressive 對應：

- 需要新 save format：anchors、local gaussians、TSDF、AnchorGraph、frame metadata。
- 需要 render/eval bridge。
- Fine-tune 不能載入全域 Gaussian；應逐 anchor 或 active-neighborhood 執行。
- Progressive 的 training 不是只有一般 per-anchor 訓練；還必須加入 overlap 區域優化，否則廣域場景在 anchor 邊界會出現不連續。

## 3. train_lod.py Additional Logic

`train_lod.py` 是 `train.py` 的擴充版，加入 runtime tracking、loss records、pose retry queue、LoD gate 與 LoD checkpoint。

### 3.1 ResourceTracker

`train_lod.py` 用 `ResourceTracker` 取代舊 `runtimes` dictionary：

```text
with tracker.track("BAB"): ...
with tracker.track("Add"): ...
with tracker.track("Init"): ...
with tracker.track("Opt"): ...
with tracker.track("anc"): ...
```

Progressive 應保留這種命名概念，但新增：

- `Track`
- `MapQueue`
- `TSDF`
- `Spawn`
- `RenderGuard`
- `LocalOpt`
- `AnchorRoll`
- `LoopVerify`
- `PGO`

### 3.2 Loss Records

`train_lod.py` 記錄每次 optimization 的平均 loss：

```text
_append_loss_record(records, stats, phase, lod, step_idx)
_save_loss_records_and_plot(records, out_dir)
```

記錄欄位：

- step
- phase
- lod
- total
- l1
- ssim
- depth

Progressive 應擴充欄位：

- tsdf
- normal
- anisotropy
- opacity_reset_count
- anchor_id
- mapping_queue_size

### 3.3 Pending Pose Retry Queue

`train_lod.py` 在 incremental pose 失敗時，若 failure reason 是 `lsf_velocity_gate`，會把 frame 放入 retry queue：

```text
pending_pose_queue.append({
  "image": image,
  "info": info,
  "desc_kpts": desc_kpts,
  "retry_count": 1,
})
```

每次成功註冊 frame 後，嘗試處理少量 pending frames。

Progressive 對應：

- retry queue 應屬於 tracking frontend。
- mapping queue 和 pose retry queue 必須分開。
- retry 成功後再 enqueue mapping task。

### 3.4 Incremental Retry Path

retry 成功時，流程和一般 incremental 類似：

```text
retry_prev_keyframes = scene_model.get_prev_keyframes(...)
Rt_retry = pose_initializer.initialize_incremental(...)
if Rt_retry is not None:
  scene_model.add_keyframe(retry_keyframe)
  scene_model.add_new_gaussians()
  scene_model.optimization_loop()
  scene_model.place_anchor_if_needed()
```

Progressive 對應：

- retry 成功後應建立 `FrameState` / local frame record。
- mapping backend 處理 spawn/fuse/opt。
- anchor rollover 仍由 controller 判斷。

### 3.5 LoD Gate

主 reconstruction save 後，`train_lod.py` 進入 LoD progression：

```text
finetune_epochs_per_level =
  max(save_at_finetune_epoch) if specified
  else 10 if lod_max > lod_min
  else 0

while current_lod_step <= lod_max:
  run finetune epochs
  lod_gate = lod_progressive_ready(...)
  while not ready and extra_epochs < max_extra:
    run extra finetune
    recheck gate

  save LoD checkpoint or completion marker

  if current_lod_step < lod_max:
    scene_model.increase_lod()
```

`lod_progressive_ready()` 目前根據 pose stability 和 projection error gate 決定是否進下一層。

Progressive 對應：

- LoD gate 不能只看全域 SceneModel。
- 應改成 per-anchor / active-neighborhood gate：
  - local pose stability
  - TSDF consistency
  - render projection stability
  - overlap RGB/depth continuity
  - loop edge confidence
- LoD increase 不應全域一次套用所有 Gaussian。

### 3.6 Single-LoD Save Fix

`train_lod.py` 對 `lod_min == lod_max` 不重複 save full scene，而是寫 marker：

```text
lod_1_complete.json
```

Progressive 對應：

- 即使 single-LoD，也需要明確 completion marker。
- marker 應包含：
  - anchors count
  - keyframes count
  - mapping queue drained
  - failure count
  - resource stats path
  - render/eval status

## 4. train.py vs train_lod.py Difference Summary

| Area | `train.py` | `train_lod.py` |
| --- | --- | --- |
| Runtime tracking | Simple `runtimes` dict | `ResourceTracker` |
| Loss record | No structured loss CSV | CSV + plot |
| Pose retry | No queue | `pending_pose_queue` for selected failures |
| LoD progression | Only optional finetune epochs | LoD gate, extra epochs, completion marker |
| Save behavior | save final scene + optional finetune saves | save final scene, LoD checkpoints/markers, resource stats |
| Failure log | pose initializer failure log | pose initializer failure log |
| Training ownership | `SceneModel` owns all | still `SceneModel` owns all |

Important: both files still rely on `SceneModel` as the central training object. `train_lod.py` adds orchestration but does not solve the global-state coupling.

## 5. SceneModel Responsibilities That Progressive Must Replace

`Progressive_train.py` cannot be complete until these old `SceneModel` responsibilities have replacements:

| Existing `SceneModel` responsibility | Progressive replacement |
| --- | --- |
| keyframe list and active frame CPU/GPU management | `ReconstructionController` + tracking frame store |
| global gaussian parameter dict | per-anchor `LocalGaussianModel` |
| `render()` / `render_from_id()` | anchor-local render adapter with `RenderGuard` |
| `add_new_gaussians()` | mapping backend Gaussian spawn policy |
| `optimization_step()` / `optimization_loop()` | active-window local optimizer |
| cross-anchor continuity from shared frames | overlap-region optimizer |
| `place_anchor_if_needed()` | controller anchor rollover + scale alignment |
| `finetune_epoch()` | per-anchor or active-neighborhood finetune |
| `save()` / `from_scene()` | new anchor-local save/load format + compatibility bridge |
| `evaluate()` | render/eval bridge over loaded anchor-local scene |

## 6. Progressive_train.py Target Flow

The new entry should follow this order:

```text
main()
  -> parse args
  -> initialize dataset and modules
  -> build ObservationBuilder
  -> build ReconstructionController
  -> build ProgressiveTrainer / mapping callback
  -> start mapping worker
  -> for frame in dataset:
       TrackingFrontend.process_frame(frame)
       if accepted keyframe:
         controller.add_frame(...)
         controller.enqueue_mapping_task(...)
       if controller.should_roll_anchor(...):
         controller.create_active_anchor(...)
       if loop candidate:
         controller.verify_and_add_loop_edge(...)
       if global graph update ready:
         controller.apply_anchor_pose_updates(...)
  -> wait for mapping queue drain
  -> stop mapping worker
  -> save reconstruction
  -> save failure log, loss records, resource stats
  -> optional render/eval
```

Key difference from `train.py` / `train_lod.py`:

- Tracking must not block on local optimization.
- Mapping must not mutate anchor pose without write lock.
- Global graph must not backpropagate through all Gaussians.
- Anchor rollover must include scale alignment.
- Training must include overlap-region optimization for adjacent anchors, not only independent local-anchor optimization.
- Opacity reset must be localized.
- TSDF must remain sparse / spatial-hash backed.

## 6.1 Overlap Optimization Requirement

Progressive training targets wide-area reconstruction. For this use case, independent local-anchor training is insufficient because each anchor can converge to a locally valid geometry/color solution while still producing visible seams at anchor boundaries.

Overlap optimization is a separate training stage after local active-window optimization:

```text
for accepted keyframe or anchor rollover:
  optimize active anchor local losses
  find neighbor anchors with co-visibility / graph adjacency
  sample overlap frames
  render active + neighbor anchor views
  optimize overlap-selected Gaussians for continuity
  record overlap residuals in loss log
```

Expected overlap losses:

- RGB consistency between anchors on shared pixels.
- Depth / inverse-depth continuity in shared views.
- TSDF surface consistency after transforming world points into each anchor-local frame.
- Scale consistency from grid-based monocular depth alignment when a new anchor is created.

Optimization boundary:

- active anchor can be optimized normally.
- neighbor anchors are only optimized for overlap-visible Gaussian subsets.
- historical anchors outside the active-neighborhood are not loaded for gradient updates.
- full global photometric optimization remains forbidden for memory reasons.

The overlap stage should produce loss log fields such as:

- `overlap_rgb`
- `overlap_depth`
- `overlap_tsdf`
- `overlap_scale`
- `overlap_num_gaussians`
- `overlap_neighbor_anchor_id`

## 7. Immediate Implementation Order

1. Create `Progressive_train.py` phase-1 entry that owns the complete training workflow and writes a Progressive manifest.
2. Create tracking frontend wrapper that reuses detector/matcher/PoseInitializer but outputs `FrameState` and pose result.
3. Create mapping callback:
   - TSDF fusion
   - Gaussian spawn
   - render guard
   - local optimization placeholder
4. Create overlap optimizer:
   - neighbor anchor selection
   - overlap frame sampling
   - overlap RGB/depth/TSDF/scale losses
   - gradient mask for neighbor overlap Gaussians
5. Create anchor-local render adapter equivalent to `SceneModel.render()`.
6. Create save/load bridge for anchor-local reconstruction.
7. Add short-sequence integration test before full dataset run.

## 8. Current Risk Points

- `PoseInitializer.initialize_incremental()` still expects old `Keyframe` objects.
- `Keyframe` currently constructs mono depth, feature maps, pose/depth/exposure optimizers together.
- `SceneModel.add_new_gaussians()` contains useful logic but is tightly coupled to global Gaussian params and old optimizer.
- Existing viewer/render scripts load `SceneModel.from_scene()`, so Progressive needs a compatibility bridge or new render path.
- The current full training script intentionally blocks new pipeline training until these interfaces exist.
