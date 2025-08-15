"""
Tests pour les nouvelles fonctionnalités de l'application.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Tests pour calculate_max_input_tokens
def test_calculate_max_input_tokens_dynamic():
    """Test de la fonction calculate_max_input_tokens avec différents modèles."""
    from core.utils import calculate_max_input_tokens
    
    # Test avec Claude 4 Sonnet
    # Contexte 200k, demande 64k sortie. Attendu: 200000 - 64000 - marge > 130000
    result_claude = calculate_max_input_tokens("claude-sonnet-4-20250514", 64000)
    assert result_claude > 130000 
    assert result_claude < 136000

    # Test avec GPT-5
    # Contexte 400k, demande 128k sortie. Attendu: 400000 - 128000 - marge > 250000
    result_gpt5 = calculate_max_input_tokens("gpt-5", 128000)
    assert result_gpt5 > 250000
    assert result_gpt5 < 272000
    
    # Test avec modèle inconnu (fallback)
    result_unknown = calculate_max_input_tokens("model-inexistant", 1000)
    assert result_unknown == 4000

def test_calculate_max_input_tokens_edge_cases():
    """Test des cas limites de calculate_max_input_tokens."""
    from core.utils import calculate_max_input_tokens
    
    # Test avec sortie très grande
    result = calculate_max_input_tokens("gpt-5", 400000)
    assert result == 0  # Pas d'espace pour l'entrée
    
    # Test avec sortie nulle
    result = calculate_max_input_tokens("gpt-4.1", 0)
    assert result > 900000  # Tout l'espace disponible

# Tests pour call_openai avec nouveaux paramètres
@patch('core.utils.OpenAI')
def test_call_openai_with_new_params(mock_openai):
    """Test de call_openai avec les nouveaux paramètres GPT-5."""
    from core.utils import call_openai
    
    # Mock de la réponse
    mock_response = MagicMock()
    mock_response.text = "Réponse de test"
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Test avec tous les paramètres
    result = call_openai(
        "gpt-5",
        "Prompt de test",
        "fake-api-key",
        temperature=0.8,
        top_p=0.9,
        max_output_tokens=2048,
        reasoning_effort="high",
        verbosity="medium"
    )
    
    assert result == "Réponse de test"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-5",
        input="Prompt de test",
        reasoning={"effort": "high"},
        text={"verbosity": "medium"}
    )

# Tests pour call_anthropic
@patch('core.utils.anthropic.Anthropic')
def test_call_anthropic_with_params(mock_anthropic):
    """Test de call_anthropic avec les paramètres."""
    from core.utils import call_anthropic
    
    # Mock de la réponse
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "Réponse Claude"
    mock_message.content = [mock_content]
    
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic.return_value = mock_client
    
    result = call_anthropic(
        "claude-3.5-sonnet-20240620",
        "Prompt de test",
        "fake-api-key",
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024
    )
    
    assert result == "Réponse Claude"
    mock_client.messages.create.assert_called_once_with(
        model="claude-3.5-sonnet-20240620",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        messages=[{"role": "user", "content": "Prompt de test"}]
    )

# Tests pour les exports
def test_export_markdown():
    """Test de l'export Markdown."""
    from core.utils import export_markdown
    
    with tempfile.TemporaryDirectory() as temp_dir:
        content = "# Test\n\nContenu de test"
        result_path = export_markdown(content, "test_section", "brouillon", temp_dir)
        
        assert os.path.exists(result_path)
        assert result_path.endswith(".md")
        
        # Vérifier le contenu
        with open(result_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == content

def test_export_docx():
    """Test de l'export DOCX."""
    from core.utils import export_docx
    
    with tempfile.TemporaryDirectory() as temp_dir:
        content = "# Test\n\nContenu de test"
        styles = {"font_family": "Arial", "font_size_body": 12}
        
        # Mock de generate_styled_docx
        with patch('core.utils.generate_styled_docx') as mock_generate:
            result_path = export_docx(content, "test_section", "final", temp_dir, styles)
            
            assert result_path.endswith(".docx")
            mock_generate.assert_called_once()

# Tests pour la configuration
def test_model_limits_config():
    """Test que la configuration MODEL_LIMITS est correcte."""
    from config_manager import MODEL_LIMITS, AVAILABLE_OPENAI_MODELS, AVAILABLE_ANTHROPIC_MODELS
    
    # Vérifier que tous les modèles listés ont des limites définies
    for model in AVAILABLE_OPENAI_MODELS + AVAILABLE_ANTHROPIC_MODELS:
        assert model in MODEL_LIMITS, f"Modèle {model} manquant dans MODEL_LIMITS"
        assert "context" in MODEL_LIMITS[model], f"Contexte manquant pour {model}"
        assert "max_output" in MODEL_LIMITS[model], f"Max output manquant pour {model}"

def test_default_params_config():
    """Test que les paramètres par défaut sont corrects."""
    from config_manager import get_config
    
    config = get_config()
    
    # Vérifier les paramètres du brouillon
    assert "reasoning_effort" in config.draft_params
    assert "verbosity" in config.draft_params
    assert config.draft_params["max_output_tokens"] > 8000
    
    # Vérifier les paramètres de la version finale
    assert "reasoning_effort" in config.final_params
    assert "verbosity" in config.final_params
    assert config.final_params["max_output_tokens"] > 16000

# Test d'intégration pour run_generation
@patch('core.utils.call_openai')
@patch('core.utils.export_markdown')
@patch('core.utils.export_docx')
def test_run_generation_integration(mock_export_docx, mock_export_md, mock_call_openai):
    """Test d'intégration de la fonction run_generation."""
    from core.utils import run_generation
    
    # Mock des fonctions
    mock_call_openai.return_value = "Texte généré"
    mock_export_md.return_value = "/tmp/test.md"
    mock_export_docx.return_value = "/tmp/test.docx"
    
    # Mock de st.session_state
    mock_session_state = {"openai_key": "fake-key"}
    
    with patch('core.utils.st.session_state', mock_session_state):
        with patch('core.utils.st.status') as mock_status:
            with patch('core.utils.st.progress') as mock_progress:
                mock_status.return_value.__enter__.return_value = mock_status
                mock_progress.return_value = MagicMock()
                
                # Test de la fonction
                result = run_generation(
                    mode="brouillon",
                    prompt="Test prompt",
                    provider="OpenAI",
                    model="gpt-5",
                    params={"max_output_tokens": 1024, "temperature": 0.7, "top_p": 0.9},
                    styles={},
                    base_name="test_section"
                )
                
                assert result[0] == "Texte généré"
                assert result[1] == "/tmp/test.md"
                assert result[2] == "/tmp/test.docx"

if __name__ == "__main__":
    pytest.main([__file__])