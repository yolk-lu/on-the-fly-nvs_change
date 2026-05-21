import json
import pathlib
import re

from Progressive_train import _collect_output_status, _parse_progressive_args, _write_manifest
from pipeline.progressive_config import parse_progressive_config


def test_progressive_config_owns_training_args():
    cfg, clean_argv = _parse_progressive_args(
        [
            "Progressive_train.py",
            "--progressive_overlap_mode",
            "reserved",
            "-s",
            "/data/scene",
            "-m",
            "/tmp/out",
            "--progressive_anchor_radius",
            "4.0",
            "--progressive_anchor_train_views",
            "3",
        ]
    )
    assert cfg.overlap_mode == "reserved"
    assert cfg.anchor_radius == 4.0
    assert cfg.anchor_train_views == 3
    assert cfg.sh_degree == 3
    assert cfg.max_active_keyframes == 200
    assert cfg.use_last_frame_proba == 0.2
    assert cfg.local_spawn_max == 16384
    assert cfg.local_spawn_target == 8192
    assert cfg.surface_sample_floor == 0.01
    assert cfg.low_frequency_spawn_fraction == 0.15
    assert cfg.edge_probability_threshold == 0.02
    assert cfg.spawn_opacity_init == 0.06
    assert cfg.max_rasterized_gaussians == 60000
    assert cfg.min_displacement == 0.03
    assert cfg.pyr_levels == 2
    assert cfg.depth_valid_epsilon == 1e-6
    assert cfg.use_vggt_pose_prior is False
    assert cfg.pose_use_lsf_velocity_gate is False
    assert clean_argv == ["Progressive_train.py"]


def test_progressive_manifest_records_reserved_overlap(tmp_path):
    cfg = parse_progressive_config(
        [
            "Progressive_train.py",
            "-s",
            "/data/scene",
            "-m",
            str(tmp_path),
            "--progressive_manifest_name",
            "manifest.json",
            "--progressive_run_label",
            "test",
        ]
    )
    (tmp_path / "metadata.json").write_text("{}")
    pcd = tmp_path / "point_clouds"
    pcd.mkdir()
    (pcd / "anchor_0.ply").write_text("ply\n")
    anchor_states = tmp_path / "anchor_states"
    anchor_states.mkdir()
    (anchor_states / "anchor_0.pt").write_bytes(b"")
    tsdf = tmp_path / "tsdf"
    tsdf.mkdir()
    (tsdf / "anchor_0.pt").write_bytes(b"")
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    (colmap / "cameras.bin").write_bytes(b"")
    (colmap / "images.bin").write_bytes(b"")
    (tmp_path / "resource_stats.txt").write_text("")
    (tmp_path / "lod_1_complete.json").write_text("{}")
    _write_manifest(str(tmp_path), cfg, "completed", started_at=1.0)
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["status"] == "completed"
    assert payload["progressive"]["overlap_optimization"] == "reserved_not_implemented"
    assert payload["progressive"]["backend_mode"] == "progressive_scene_model"
    assert payload["progressive"]["trainer_entrypoint"] == "Progressive_train.py"
    assert payload["progressive"]["pose_graph_optimization"] == "sim3_enabled_after_verified_loop"
    assert payload["progressive_config"]["pose"]["use_vggt_pose_prior"] is False
    assert payload["progressive_config"]["pose"]["pose_use_lsf_velocity_gate"] is False
    assert payload["outputs"]["complete"]


def test_output_status_reports_missing_required_files(tmp_path):
    status = _collect_output_status(str(tmp_path))
    assert not status["complete"]
    assert "metadata_json" in status["missing"]
    assert "anchor_ply" in status["missing"]


def test_progressive_train_no_longer_imports_legacy_scene_store():
    source = open("Progressive_train.py").read()
    assert "from args import get_args" not in source
    assert "from scene.scene_model import SceneModel" not in source
    assert "from scene.keyframe import Keyframe" not in source
    assert "self.scene_model" not in source


def test_progressive_config_exposes_args_used_by_progressive_modules():
    cfg = parse_progressive_config(["Progressive_train.py", "-s", "/data/scene", "-m", "/tmp/out"])
    files = [pathlib.Path("Progressive_train.py")]
    for root in ("pipeline", "poses", "scene", "dataloaders"):
        files.extend(pathlib.Path(root).rglob("*.py"))
    attrs = set()
    for path in files:
        attrs.update(re.findall(r"(?:self\.)?args\.([A-Za-z_][A-Za-z0-9_]*)", path.read_text(errors="ignore")))
    missing = sorted(attr for attr in attrs if not hasattr(cfg, attr))
    assert missing == []
