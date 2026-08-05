# scripts 目录结构

实际脚本按用途分目录；根目录不保留重复入口，避免 agent 误跑旧路线。

- `data/`: `download_data.sh`, `onboard_object.py`, `rebuild_bank_fixed_views.py`
- `eval/`: `run_linemod.py`, `run_speed.py`, `summarize13.py`, `run_ablation.py`
- `analysis/`: `extract_matches.py`, `calib_area_bias.py`, `diag_dclass.py`, `diag_tz_loss.py`, `verify_align_select.py`, `verify_anchor_depth.py`, `verify_gtmask_tz.py`
- `maintenance/`: `patch_depth_anchor_maps.py`, `extract13_dc2.sh`, `extract_dc2_fix.sh`, `rerun13_bg0.sh`
