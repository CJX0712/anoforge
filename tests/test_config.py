import pytest

from core.config import Config, load_config
from core.errors import ConfigError


def test_config_defaults():
    c = Config()
    assert 0 < c.contamination < 1
    assert c.diversity_weight >= 0


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ANOFORGE_CONTAMINATION", "0.2")
    monkeypatch.setenv("ANOFORGE_SEED", "42")
    monkeypatch.setenv("ANOFORGE_DIVERSITY_WEIGHT", "2.5")
    c = Config.from_env()
    assert c.contamination == 0.2
    assert c.seed == 42
    assert c.diversity_weight == 2.5


def test_config_invalid_contamination(monkeypatch):
    monkeypatch.setenv("ANOFORGE_CONTAMINATION", "1.5")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_validate():
    c = Config(contamination=0.5)
    c.contamination = 1.5
    with pytest.raises(ConfigError):
        c.validate()


def test_load_config_fallback():
    c = load_config()
    assert isinstance(c, Config)
