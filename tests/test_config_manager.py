import sys
from pathlib import Path

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_manager import get_config, AppConfig

def test_config_loading():
    """Test de base pour le chargement de la configuration."""
    config = get_config()
    assert isinstance(config, AppConfig)
    assert hasattr(config, 'drafter_model')
    assert hasattr(config, 'refiner_model')
