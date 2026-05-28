# SLAM Front-End Issue Report

## 1. [HIGH] 軌跡出現明顯跳點（pose jump）
- Evidence: jump_ratio=0.143, step_max/step_median=234931.20
- Likely Root Cause: 2D-3D對應包含錯配仍被PnP/MiniBA接受，或低inlier fallback門檻過寬。
- Actionable Fix: 提高 `--min_num_inliers`、收緊低inlier接受條件，並在 `initialize_incremental()` 加入『若與軌跡先驗差異過大則拒絕該pose』的硬閥值。

## 2. [MEDIUM] 相機方向變化呈鋸齒型（zig-zag）
- Evidence: turn_spike_ratio=0.376, turn_p95=170.1deg
- Likely Root Cause: 相鄰幀匹配幾何約束不足，增量位姿在局部最小值間震盪。
- Actionable Fix: 在前端加入短窗平滑先驗（constant velocity/acceleration）並對大角度轉折施加懲罰；同時提高 match 幾何一致性檢查。

## 3. [MEDIUM] 俯視相機光軸不穩定
- Evidence: forward_dev_deg_p95=53.5deg
- Likely Root Cause: 位姿初始化對 pitch/roll 的幾何約束不足，導致姿態在弱紋理區抖動。
- Actionable Fix: 在初始化與優化加入俯視姿態先驗（限制 pitch/roll），先驗可做成可開關參數避免影響一般場景。

