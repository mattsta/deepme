"""set_default_profile must never lose other config.toml sections."""

import tomllib

from deepme import config, profiles


def test_set_default_preserves_all_sections(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[server]\nhost = "1.2.3.4"\nuser = "ubuntu"\nport = 8000\n\n'
        "[cluster]\ngpus = 8\nvram_gb = 32\n\n"
        "[policy]\nmin_gguf_bits = 6\n"
    )
    monkeypatch.setattr(config, "CONFIG_FILE", cfg)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(profiles, "list_profiles", lambda: ["some-model"])
    profiles.set_default_profile("some-model")
    result = tomllib.loads(cfg.read_text())
    assert result["server"]["host"] == "1.2.3.4"
    assert result["server"]["port"] == 8000
    assert result["cluster"] == {"gpus": 8, "vram_gb": 32}
    assert result["policy"] == {"min_gguf_bits": 6}
    assert result["model"]["default"] == "some-model"
