# Progressive Anchor-Local Execution Flow

本文記錄 Progressive / anchor-local pipeline 的目標執行流程、目前可執行檢查，以及後續正式訓練入口需要補上的接線責任。

## 目前狀態

- `pipeline/reconstruction_controller.py` 已是 Tracking / Mapping / Global Graph 的邊界協調層。
- `pipeline/anchor_local_train.py --smoke-only` 只作模組 smoke，不是完整訓練入口。
- `Progressive_train.py` 是正式訓練入口的 phase 1。
- phase 1 已改為 Progressive 自己持有完整訓練 loop，不再呼叫 `train.py` 或 `train_lod.py`；overlap optimization 只保留 hook，不啟用。
- anchor-local render / optimizer / save-load bridge 仍會逐步替換 phase 1 backend。

## 高階執行流程

```text
Input image stream
  -> ObservationBuilder
  -> Tracking frontend
       - feature extraction
       - feature matching
       - ParallaxBA pose initialization / refinement
       - keyframe decision
       - enqueue MappingTask
  -> Mapping backend thread
       - consume keyframe buffer
       - spawn local Gaussians
       - fuse depth into anchor-local TSDF
       - render guarded active window
       - optimize local Gaussian + pose/depth/exposure states
       - optimize anchor overlap regions
       - localized opacity reset
       - anchor rollover if needed
  -> Global map layer
       - maintain AnchorGraph
       - verify loop closure candidates
       - pose graph update
       - rigid anchor pose correction with covariance rotation
  -> Save / render / evaluate
```

## 1. Preflight

1. Parse training args and paths.
2. Initialize deterministic seeds and CUDA memory policy.
3. Initialize:
   - image dataset or stream dataset
   - `Detector`
   - `Matcher`
   - Parallax pose initializer
   - `DenseExtractor`
   - `MonoDepthEstimator`
   - `ObservationBuilder`
   - `ReconstructionController`
4. Start mapping worker:

```python
controller = ReconstructionController(mapping_callback=mapping_step)
controller.start_mapping_worker()
```

`mapping_callback` is the backend implementation that consumes `MappingTask`. It must not block tracking longer than queue insertion.

## 2. Tracking Frontend

Tracking is the high-frequency loop. It must not wait for local Gaussian optimization.

Per frame:

1. Load image and metadata.
2. Build `FrameState`:
   - RGB image
   - feature keypoints/descriptors
   - dense feature map if enabled
   - monocular inverse depth and confidence
3. Match against previous keyframe or selected local keyframes.
4. Run ParallaxBA pose initialization:
   - bootstrap phase uses multiple initial frames
   - incremental phase uses previous active keyframes
5. Decide keyframe insertion by displacement, inlier count, test frame flag, and pose validity.
6. If keyframe is accepted:
   - add frame to controller
   - push `MappingTask(frame, anchor_id, kind="keyframe")`
   - continue tracking next frame without waiting for backend optimization

Queue failure policy:

- If mapping queue is full, frontend should record a failure reason and either drop non-critical mapping work or downgrade the frame to tracking-only.
- Test/evaluation frames should preserve pose metadata even if mapping is skipped.

## 3. Mapping Backend

Mapping runs in a separate thread and consumes `MappingTask`.

Backend step for each keyframe:

1. Acquire the target `AnchorLocalMap`.
2. Fuse monocular depth into that anchor's local TSDF:
   - points are transformed from camera to world, then world to anchor-local coordinates
   - `AdaptiveTSDF` stores samples through spatial hash keys, not dense 3D arrays
3. Spawn local Gaussians:
   - LoG direct sampling from image high-frequency regions
   - DepthAnything / MonoDepth confidence gate
   - optional render-depth occlusion gate
   - initialize SH color, opacity, scale, rotation
4. Run render guard before rasterization:
   - finite tensor check
   - depth / distance check
   - projected screen-size check
5. Optimize active window:
   - photometric loss
   - SSIM loss
   - monocular depth loss
   - TSDF surface loss
   - TSDF normal alignment loss
   - Gaussian anisotropy regularization
6. Optimize overlap regions when the active anchor has neighbor anchors:
   - sample frames that observe both anchors
   - render active and neighbor anchors in the overlap field of view
   - enforce color / depth / TSDF consistency across the shared boundary
   - restrict gradients to the active anchor and explicitly selected neighbor overlap Gaussians
   - do not load or optimize the full global map
7. Apply localized opacity reset:
   - no global opacity reset
   - only visible, low-view-diversity, high-gradient, TSDF-unstable Gaussians can be reset

The backend must never mutate an anchor pose or covariance without taking the anchor write lock.

Overlap optimization is mandatory for wide-area scenes. General per-anchor training can make each local map look correct alone, but it does not guarantee continuity at anchor boundaries. The overlap stage is where Progressive differs from ordinary local training.

## 4. Anchor Lifecycle

The active anchor owns high-frequency local optimization state. Older anchors are sealed and moved out of the active GPU window.

Rollover check:

```python
controller.should_roll_anchor(
    cam_centre_world,
    max_anchor_radius=...,
    min_keyframes=...,
)
```

When rollover triggers:

1. Stop adding new Gaussians to the current active anchor.
2. Seal current anchor:

```python
old_anchor_id = controller.seal_active_anchor()
```

3. Compute inter-anchor monocular scale alignment:
   - use overlapping frames between old and new anchor
   - estimate grid-based inverse-depth ratio
   - store the scale alignment result on the new anchor
4. Create new active anchor:

```python
new_anchor = controller.create_active_anchor(
    R_anchor_to_world,
    t_anchor_to_world,
    reference_frame=last_old_anchor_frame,
    new_frame=current_frame,
)
```

5. Add a sequential edge in `AnchorGraph`.
6. Update active GPU set through `AnchorChunkManager`.

Memory rule:

- GPU should hold only the active anchor and a small neighbor set.
- Inactive anchors are moved to CPU and represented globally by pose, bounds, graph node, and metadata.
- Neighbor anchors needed for overlap optimization may be temporarily loaded, optimized only on overlap-selected Gaussians, then offloaded again.

## 4.1 Overlap Optimization

Overlap optimization handles visual and geometric continuity between adjacent anchors.

Trigger conditions:

- new anchor is created from an old active anchor
- active camera observes an existing neighbor anchor
- loop closure candidate is accepted and creates a new graph edge
- render/eval detects a boundary discontinuity near an anchor transition

Inputs:

- active anchor
- one or more neighbor anchors
- overlapping keyframes or current frame with co-visibility
- active and neighbor TSDF queries
- render packages from the same camera pose

Loss terms:

- overlap photometric consistency: rendered RGB from active/neighbor anchors should agree in shared visible pixels
- overlap depth consistency: rendered inverse depth should be continuous across the boundary
- overlap TSDF consistency: local TSDF zero-crossing should not jump after world-to-anchor transforms
- scale consistency: use grid-based mono-depth scale alignment when the overlap comes from a new anchor rollover

Gradient boundary:

- optimize the active anchor normally
- optimize neighbor anchor only for Gaussians selected by overlap visibility
- never optimize all historical anchors at once
- never let overlap optimization bypass `RenderGuard`

Success signal:

- overlap RGB/depth residual decreases or stays finite
- no non-finite Gaussian parameters
- no new renderer CUDA error
- anchor boundary render does not show a visible discontinuity in the overlap view set

## 5. Global Graph And Loop Closure

The global layer only optimizes anchor poses. It must not backpropagate photometric loss through all Gaussians.

Loop candidate creation:

1. Find candidate anchors by bounding box or spatial overlap.
2. Sample representative keyframes from both anchors.
3. Run feature matching through LightGlue / MASt3R-compatible matcher callback.
4. Verify homography or geometric consistency:

```python
result = controller.verify_and_add_loop_edge(
    src_anchor_id,
    dst_anchor_id,
    left_frame,
    right_frame,
    matcher_fn,
    T_src_to_dst,
)
```

Only accepted loop edges are inserted into `AnchorGraph`.

After pose graph optimization:

```python
controller.apply_anchor_pose_updates(updates)
```

This method must:

- acquire write lock per anchor
- update anchor 6-DoF pose
- rotate Gaussian covariance by rigid correction
- keep local Gaussian and TSDF coordinates anchor-relative

## 6. Thread Safety Contract

| Operation | Lock |
| --- | --- |
| Read anchor pose for tracking/render | anchor read lock |
| Append keyframe id to anchor | anchor write lock |
| Seal/offload anchor | anchor write lock |
| Apply PGO pose update | anchor write lock |
| Covariance rotation | anchor write lock |
| Mapping queue push | queue internal lock |
| Mapping task consumption | queue internal lock |

Tracking should only take short read locks. Mapping and global correction may take write locks.

## 7. Current Executable Checks

Run all current unit/module checks:

```bash
scripts/run_plan_unit_tests.sh
```

Expected current result:

```text
pytest: all plan targets pass
module smoke: finite TSDF / anchor / render guard counts
```

Run module smoke only:

```bash
python -m pipeline.anchor_local_train --smoke-only
```

Check planned flow text:

```bash
scripts/run_plan_flow.sh
```

## 8. Full Training Gate

Full Progressive training phase 1 is complete when `Progressive_train.py` can finish its own training workflow and write `progressive_manifest.json`.

Full anchor-local Progressive training is not complete until `Progressive_train.py` provides:

- Tracking frontend loop
- Async mapping backend callback
- anchor-local differentiable render adapter
- active-window local training loop
- overlap-region optimizer for adjacent anchors
- save/load format for anchor-local maps, TSDF, AnchorGraph, and keyframes
- render/eval bridge
- failure log and resource stats

Completion condition:

- Aerial / wide-area ordered image sequence runs to completion.
- Reconstruction is saved in the new format.
- Render/eval can load the saved reconstruction.
- Adjacent anchor overlap regions render continuously without visible seams.
- No renderer CUDA illegal memory access.
- If reconstruction fails, failure reason is written explicitly.

Phase 1 intentionally prioritizes end-to-end train completion. It is not the final anchor-local implementation until the Progressive-owned SceneModel backend is replaced by the anchor-local render / optimizer / save-load bridge.
