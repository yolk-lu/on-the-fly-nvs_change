# Overlap Joint Optimization 實作規格

## 1. 目標

Progressive pipeline 目前的 anchor 是各自進行 local optimization。對 MatrixCity 這類無人機廣域場景，單獨訓練每個 anchor 會在 anchor 邊界產生幾何與顏色不連續。Overlap 的目標是在 anchor rollover 後，對 Active Anchor 與鄰近 Anchor 的共同觀測區域進行局部 joint optimization，讓跨 anchor 的 Gaussian 分布、render depth、顏色與尺度保持連續。

本文件只規劃 `Progressive_train.py` / `ProgressiveSceneModel` 新架構，不回接舊 `train.py`、`train_lod.py`、`SceneModel` 或 viewer。

## 2. 核心概念

令當前 Active Anchor 為 `A0`，候選鄰近 Anchor 為 `Aj`，候選集合為 `N(A0)`。

Overlap 不是固定每 N 張 keyframe 執行，而是根據下列指標判斷是否值得執行：

- `C_DINO(A0, Aj)`：兩個 anchor 內 keyframe 的最高 DINO/VLAD global descriptor cosine similarity。
- `C_co(A0, Aj)`：兩個 anchor 之間 local feature matching 的共視幾何強度。
- `Delta_motion`：從上一個 overlap / rollover 到目前累積的 camera motion gain。
- `theta_parallax`：共視 matches 的 median parallax angle。

最終觸發分數：

```text
Phi(A0, Aj)
  = 0.35 * C_DINO
  + 0.30 * min(C_co / 200, 1)
  + 0.20 * (1 - exp(-Delta_motion / 1.0))
  + 0.15 * min(theta_parallax / 10deg, 1)
```

觸發條件：

```text
Trigger = max_j Phi(A0, Aj) >= tau_trigger
```

預設：

```text
tau_trigger = 0.55
min_dino = 0.50
min_coview_matches = 80
```

只有當 `Phi >= tau_trigger` 且滿足 `C_DINO >= min_dino` 或 `C_co >= min_coview_matches` 時，才建立 overlap event。

## 3. 新增模組

### 3.1 `pipeline/overlap_manager.py`

新增 `OverlapManager`，負責：

- 從 `controller.anchors` 找 `N(A0)`。
- 從 `place_index.records` 取得 frame-level DINO/VLAD descriptor。
- 用 top DINO keyframe pairs 執行 local matcher，估計 `C_co`。
- 用 keyframe pose 計算 `Delta_motion` 與 `theta_parallax`。
- 產生 `OverlapCandidate` 與 `OverlapEvent`。
- 呼叫 trainer callback 執行 handoff 與 joint optimization。

建議 dataclass：

```python
@dataclass
class OverlapCandidate:
    src_anchor_id: int
    dst_anchor_id: int
    c_dino: float
    c_coview: float
    delta_motion: float
    theta_parallax_deg: float
    phi: float
    src_keyframe_indices: list[int]
    dst_keyframe_indices: list[int]
    triggered: bool
    reason: str


@dataclass
class OverlapEvent:
    src_anchor_id: int
    dst_anchor_id: int
    triggered: bool
    c_dino: float
    c_coview: float
    delta_motion: float
    theta_parallax_deg: float
    phi: float
    handoff_count: int
    num_views: int
    loss_before: float
    loss_after: float
    reason: str
```

### 3.2 `ProgressiveTrainer` 接線

在 `ProgressiveTrainer.initialize()` 建立：

```python
self.overlap_manager = OverlapManager(...)
```

在 `_attach_frame_to_controller(...)` 中：

```text
add frame to active anchor
update place_index
check loop closure
run mapping step
check overlap trigger
```

Overlap 檢查應放在 mapping step 後，因為新的 keyframe 需要先完成 spawn / local optimization，才有足夠 Gaussian 可用於 handoff 與 continuity loss。

## 4. Candidate Selection

`N(A0)` 包含：

- sequential previous anchor：`A0.anchor_id - 1`
- `controller.update_active_set(camera_center)` 回傳的 active anchors
- bbox / proximity fallback anchors
- loop closure accepted anchors

排除條件：

- `Aj == A0`
- anchor Gaussian 數量為 0
- `Aj` 沒有可用 keyframes
- anchor 不可搬到 renderer device，或 device transfer 失敗

DINO keyframe pair 選擇：

```text
for Ka in keyframes(A0):
  for Kb in keyframes(Aj):
    S_DINO = cosine(desc(Ka), desc(Kb))
取 top K pairs，預設 K=6
C_DINO = max(S_DINO)
```

`C_co` 計算：

```text
C_co = sum(num_inlier_matches(Ka, Kb)) for top DINO pairs
```

第一版可用 `observation_builder.match(..., remove_outliers=True)` 的 inlier matches。若 matcher 沒有回傳 inlier mask，則使用 `remove_outliers=False` 的 match count，但 metadata 必須標記 `coview_mode=raw_matches`。

## 5. Handoff

Overlap 觸發後，先從 `Aj` 複製一部分高可信 boundary Gaussians 到 `A0`，讓新 anchor 不是從空白邊界開始優化。

選點條件：

```text
opacity = sigmoid(anchor_j.opacity)
finite xyz/scaling/rotation/color
opacity >= overlap_handoff_min_opacity
world point within A0 overlap radius OR visible in overlap keyframes
```

預設：

```text
overlap_handoff_min_opacity = 0.10
overlap_handoff_max_gaussians = 30000
overlap_handoff_radius_factor = 1.25
```

轉換流程：

```text
xyz_world = Aj.local_to_world(xyz_j)
xyz_a0 = A0.world_to_local(xyz_world)
```

Gaussian scale：

```text
world_scale = exp(scaling_j) * Aj.s_anchor_to_world
local_scale_a0 = world_scale / A0.s_anchor_to_world
scaling_a0 = log(local_scale_a0)
```

Gaussian rotation：

```text
R_world = Aj.R_anchor_to_world @ R_local_j
R_local_a0 = A0.R_anchor_to_world.T @ R_world
```

顏色、opacity、SH coefficients 直接複製。第一版不從 `Aj` 刪除點，避免破壞已封存 anchor。

## 6. Joint Optimization

新增：

```python
ProgressiveSceneModel.optimization_step_overlap(
    anchor_ids: list[int],
    view_items: list[tuple[TrackingKeyframe, FrameState]],
    c_dino: float,
    n_iters: int,
) -> dict | None
```

和現有 `optimization_step_multiview(...)` 的差異：

- render 使用 `active_anchor_ids=[A0, Aj]`
- optimizer 同時包含 `A0` 與 `Aj` 的 Gaussian parameters
- loss 增加 `L_geo_continuity`
- 不更新 keyframe pose
- 不處理 test frames

總 loss：

```text
L_overlap_total =
    L_photo
  + L_depth
  + C_DINO * lambda_overlap_geo * L_geo_continuity
  + lambda_anisotropy * L_anisotropy
```

`L_geo_continuity`：

```text
P0 = high-opacity finite Gaussians from A0 in world space
Pj = high-opacity finite Gaussians from Aj in world space
L_geo_continuity = chamfer(P0, Pj)
```

採樣限制：

```text
overlap_geo_max_points = 4096
```

若任一 anchor 的 valid points 少於 `overlap_geo_min_points=128`，則 `L_geo_continuity=0`，並在 event metadata 記錄 `geo_continuity_skipped=too_few_points`。

## 7. View Sampling

Overlap views 由兩部分組成：

```text
new_views = recent non-test keyframes from A0
old_views = DINO/top-match keyframes from Aj
```

預設：

```text
overlap_new_keyframes = 3
overlap_prev_keyframes = 6
overlap_max_views = 8
```

view selection 規則：

1. 先加入觸發 overlap 的 current keyframe。
2. 加入 `A0` 最近非 test keyframes。
3. 加入 top DINO pair 中的 `Aj` keyframes。
4. 移除缺 frame state、缺 keyframe、或 `is_test=True` 的項目。
5. 若最後 view 數小於 2，跳過 overlap event。

## 8. Config

新增到 `pipeline/progressive_config.py`：

```text
progressive_overlap_mode: enabled | reserved | off
progressive_overlap_trigger_threshold = 0.55
progressive_overlap_min_dino = 0.50
progressive_overlap_min_coview_matches = 80
progressive_overlap_joint_iterations = 10
progressive_overlap_new_keyframes = 3
progressive_overlap_prev_keyframes = 6
progressive_overlap_max_views = 8
progressive_overlap_geo_weight = 1.0
progressive_overlap_geo_max_points = 4096
progressive_overlap_geo_min_points = 128
progressive_overlap_handoff_min_opacity = 0.10
progressive_overlap_handoff_max_gaussians = 30000
progressive_overlap_handoff_radius_factor = 1.25
```

相容性：

- `reserved` 視為 `enabled`，但 manifest 中要記錄原始輸入值。
- `off` 完全不建立 manager、不執行 trigger、不做 handoff。

## 9. Metadata / Diagnostics

`metadata.json` 與 `progressive_manifest.json` 需要記錄：

```json
{
  "overlap_optimization": "enabled",
  "overlap_events": [
    {
      "src_anchor_id": 6,
      "dst_anchor_id": 5,
      "triggered": true,
      "c_dino": 0.71,
      "c_coview": 143,
      "delta_motion": 1.34,
      "theta_parallax_deg": 6.2,
      "phi": 0.63,
      "handoff_count": 12000,
      "num_views": 7,
      "loss_before": 0.42,
      "loss_after": 0.31,
      "reason": "triggered"
    }
  ]
}
```

`loss_records.csv` 新增欄位：

```text
overlap_total
overlap_photo
overlap_depth
overlap_geo
overlap_anisotropy
overlap_c_dino
overlap_phi
overlap_src_anchor
overlap_dst_anchor
overlap_handoff_count
```

## 10. Test Plan

### 10.1 Trigger tests

- `C_DINO` 高但 `C_co` 低，且 `Phi < threshold` 時不觸發。
- `C_co` 高或 `Phi >= threshold` 時觸發。
- sequential previous anchor 不被 loop closure 的 anchor gap 排除。
- bbox/proximity fallback 能產生候選。

### 10.2 Handoff tests

- `Aj local -> world -> A0 local` 後，world position 保持一致。
- `Aj.s_anchor_to_world != A0.s_anchor_to_world` 時，world covariance 尺度保持一致。
- low opacity、non-finite、超過 cap 的 Gaussians 不會被 handoff。

### 10.3 Joint optimization tests

- overlap optimizer 同時更新 `A0` 與 `Aj` optimizer parameters。
- `L_geo_continuity` 對接近點低、分離點高。
- `overlap_geo_max_points` 會限制 Chamfer 計算點數，避免 OOM。
- test frames 不會進入 overlap optimizer。

### 10.4 Pipeline tests

- `progressive_overlap_mode=off` 時沒有 overlap event。
- `reserved` / `enabled` 時 manager 建立，manifest 不再寫 `reserved_not_implemented`。
- 短流程至少產生 rejected 或 triggered overlap diagnostics。
- test render metadata 可記錄當次 active overlap anchors。

## 11. Implementation Order

1. 新增 overlap config 與 parser tests。
2. 新增 `OverlapManager` dataclasses 與 trigger score tests。
3. 實作 DINO keyframe pair query 與 `C_DINO`。
4. 實作 local matcher `C_co` 與 parallax / motion metrics。
5. 實作 handoff transform。
6. 實作 `optimization_step_overlap(...)` 與 `L_geo_continuity`。
7. 接入 `ProgressiveTrainer._attach_frame_to_controller(...)`。
8. 補 metadata、loss records、manifest。
9. 跑短流程檢查 boundary test render。

## 12. 目前不做

- 不更新 keyframe pose。
- 不做 full global photometric optimization。
- 不在 overlap 裡直接觸發 PGO。
- 不回接 viewer。
- 不刪除舊 anchor Gaussians。
