import json

from Progressive_train import ProgressiveRunConfig, _parse_progressive_args, _write_manifest


def test_progressive_args_are_stripped_before_delegated_trainer():
    cfg, trainer_argv = _parse_progressive_args(
        [
            "Progressive_train.py",
            "--progressive_overlap_mode",
            "reserved",
            "-s",
            "/data/scene",
            "-m",
            "/tmp/out",
            "--num_iterations",
            "1",
        ]
    )
    assert cfg.overlap_mode == "reserved"
    assert "--progressive_overlap_mode" not in trainer_argv
    assert trainer_argv == ["Progressive_train.py", "-s", "/data/scene", "-m", "/tmp/out", "--num_iterations", "1"]


def test_progressive_manifest_records_reserved_overlap(tmp_path):
    cfg = ProgressiveRunConfig(
        delegated_trainer="train_lod.py",
        overlap_mode="reserved",
        backend_mode="legacy_scene_model",
        manifest_name="manifest.json",
        run_label="test",
    )
    _write_manifest(str(tmp_path), cfg, "completed", started_at=1.0)
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["status"] == "completed"
    assert payload["progressive"]["overlap_optimization"] == "reserved_not_implemented"
    assert payload["progressive"]["backend_mode"] == "legacy_scene_model"
