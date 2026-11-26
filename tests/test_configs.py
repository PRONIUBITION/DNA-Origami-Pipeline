from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TYPES = {"0": 2, "1": 2, "00": 3, "01": 3, "10": 3, "11": 3}


def test_monomer_type_configs():
    config_dir = ROOT / "monomer" / "stage2_hrnet" / "configs" / "types"
    for monomer_type, keypoints in TYPES.items():
        config = yaml.safe_load(
            (config_dir / f"type_{monomer_type}.yaml").read_text(encoding="utf-8")
        )
        assert config["data"]["num_keypoints"] == keypoints
        assert config["model"]["num_heatmaps"] == keypoints


def test_dimer_configs_parse():
    config_dir = ROOT / "dimer" / "stage2_hrnet" / "configs"
    for path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "data" in config
        assert "model" in config
