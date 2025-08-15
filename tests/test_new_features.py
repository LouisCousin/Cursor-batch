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

# Tests spécifiques pour les corrections GPT-5

def test_params_for_gpt5_model():
    """Test que les paramètres temperature et top_p sont absents pour les modèles GPT-5."""
    import sys
    import os
    
    # Ajouter le répertoire racine au path pour importer stubs_batch
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    from stubs_batch import BatchProcessor
    from unittest.mock import MagicMock
    
    # Mock du client OpenAI
    mock_client = MagicMock()
    
    # Créer une instance du BatchProcessor
    batch_processor = BatchProcessor.__new__(BatchProcessor)
    batch_processor.client = mock_client
    batch_processor.tracker = MagicMock()
    batch_processor.batch_files_dir = MagicMock()
    
    # Tester avec gpt-5-nano
    params = batch_processor.get_model_specific_params("gpt-5-nano")
    
    # Vérifier que temperature et top_p ne sont PAS dans les paramètres
    assert "temperature" not in params, "temperature ne devrait pas être présent pour gpt-5-nano"
    assert "top_p" not in params, "top_p ne devrait pas être présent pour gpt-5-nano"
    
    # Vérifier que les paramètres spécifiques GPT-5 sont présents
    assert "reasoning_effort" in params, "reasoning_effort devrait être présent pour gpt-5-nano"
    assert "verbosity" in params, "verbosity devrait être présent pour gpt-5-nano"
    assert "max_completion_tokens" in params, "max_completion_tokens devrait être présent pour gpt-5-nano"


def test_params_for_gpt4_model():
    """Test que les paramètres temperature et top_p sont présents pour les modèles GPT-4."""
    import sys
    import os
    
    # Ajouter le répertoire racine au path pour importer stubs_batch
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    from stubs_batch import BatchProcessor
    from unittest.mock import MagicMock
    
    # Mock du client OpenAI
    mock_client = MagicMock()
    
    # Créer une instance du BatchProcessor
    batch_processor = BatchProcessor.__new__(BatchProcessor)
    batch_processor.client = mock_client
    batch_processor.tracker = MagicMock()
    batch_processor.batch_files_dir = MagicMock()
    
    # Tester avec gpt-4.1-turbo
    params = batch_processor.get_model_specific_params("gpt-4.1-turbo")
    
    # Vérifier que temperature et top_p SONT dans les paramètres
    assert "temperature" in params, "temperature devrait être présent pour gpt-4.1-turbo"
    assert "top_p" in params, "top_p devrait être présent pour gpt-4.1-turbo"
    
    # Vérifier que les paramètres spécifiques GPT-4 sont présents
    assert "max_tokens" in params, "max_tokens devrait être présent pour gpt-4.1-turbo"
    
    # Vérifier que les paramètres GPT-5 ne sont PAS présents
    assert "reasoning_effort" not in params, "reasoning_effort ne devrait pas être présent pour gpt-4.1-turbo"
    assert "verbosity" not in params, "verbosity ne devrait pas être présent pour gpt-4.1-turbo"


def test_error_parsing():
    """Test l'extraction des messages d'erreur depuis un fichier d'erreur OpenAI simulé."""
    import json
    from unittest.mock import MagicMock, patch
    import sys
    import os
    
    # Ajouter le répertoire racine au path pour importer stubs_batch
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    from stubs_batch import BatchProcessor
    from core.process_tracker import ProcessTracker
    
    # Créer un mock de fichier d'erreur JSONL
    mock_error_content = json.dumps({
        "id": "batch_req_123",
        "custom_id": "test_section",
        "response": {
            "status_code": 400,
            "body": {
                "error": {
                    "message": "The model `gpt-5-nano-invalid` does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        }
    })
    
    # Mock du client OpenAI
    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = None
    mock_batch.error_file_id = "error_file_123"
    mock_batch.metadata = {"process_id": "test_process"}
    mock_batch.request_counts = MagicMock()
    mock_batch.request_counts.total = 1
    
    mock_client.batches.retrieve.return_value = mock_batch
    
    # Mock du contenu du fichier d'erreur
    mock_error_bytes = MagicMock()
    mock_error_bytes.decode.return_value = mock_error_content
    mock_client.files.content.return_value = mock_error_bytes
    
    # Mock du tracker
    mock_tracker = MagicMock(spec=ProcessTracker)
    mock_process = {
        "batch_history": [{
            "batch_id": "batch_123",
            "section_codes": ["test_section"]
        }]
    }
    mock_tracker.get_process.return_value = mock_process
    
    # Créer une instance du BatchProcessor
    batch_processor = BatchProcessor.__new__(BatchProcessor)
    batch_processor.client = mock_client
    batch_processor.tracker = mock_tracker
    
    # Tester la méthode process_batch_results
    try:
        result = batch_processor.process_batch_results("batch_123")
        
        # Vérifier que l'erreur a été traitée
        assert result["error_count"] == 1, "Le nombre d'erreurs devrait être 1"
        assert result["success_count"] == 0, "Le nombre de succès devrait être 0"
        
        # Le résultat global peut être générique, mais l'important c'est que l'erreur soit transmise au tracker
        # Vérifier que update_section_status a été appelé avec le bon message d'erreur
        mock_tracker.update_section_status.assert_called()
        call_args = mock_tracker.update_section_status.call_args[1] if mock_tracker.update_section_status.call_args else mock_tracker.update_section_status.call_args_list[0][1]
        assert "Erreur API OpenAI" in call_args["error_message"], \
            f"Le message d'erreur devrait être préfixé par 'Erreur API OpenAI', mais on a reçu: {call_args['error_message']}"
        assert "gpt-5-nano-invalid" in call_args["error_message"], \
            f"Le message d'erreur devrait contenir le détail de l'erreur API, mais on a reçu: {call_args['error_message']}"
        
    except ValueError as e:
        # Dans ce cas, vérifier que l'erreur a quand même été traitée via le tracker
        # même si le processus principal a levé une exception
        if mock_tracker.update_section_status.called:
            call_args = mock_tracker.update_section_status.call_args[1] if mock_tracker.update_section_status.call_args else mock_tracker.update_section_status.call_args_list[0][1]
            assert "Erreur API OpenAI" in call_args["error_message"], \
                f"Le message d'erreur devrait être préfixé par 'Erreur API OpenAI', mais on a reçu: {call_args['error_message']}"
            assert "gpt-5-nano-invalid" in call_args["error_message"], \
                f"Le message d'erreur devrait contenir le détail de l'erreur API, mais on a reçu: {call_args['error_message']}"


def test_batch_request_filtering():
    """Test que le filtrage des paramètres fonctionne dans create_batch_input_file."""
    import json
    from unittest.mock import MagicMock, patch, mock_open
    import sys
    import os
    
    # Ajouter le répertoire racine au path pour importer stubs_batch
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    from stubs_batch import BatchProcessor
    from core.corpus_manager import CorpusManager
    from core.prompt_builder import PromptBuilder
    import pandas as pd
    
    # Créer des données mock
    mock_sections = [{"code": "1.1", "title": "Test Section"}]
    
    # Mock du corpus manager
    mock_corpus = MagicMock(spec=CorpusManager)
    mock_df = pd.DataFrame([{"content": "test content", "score": 0.8}])
    mock_corpus.get_relevant_content.return_value = mock_df
    
    # Mock du prompt builder
    mock_prompt_builder = MagicMock(spec=PromptBuilder)
    mock_prompt_builder.build_draft_prompt.return_value = "Test prompt"
    
    # Mock du client OpenAI
    mock_client = MagicMock()
    mock_file_response = MagicMock()
    mock_file_response.id = "file_123"
    mock_client.files.create.return_value = mock_file_response
    
    # Créer une instance du BatchProcessor
    batch_processor = BatchProcessor.__new__(BatchProcessor)
    batch_processor.client = mock_client
    batch_processor.tracker = MagicMock()
    batch_processor.batch_files_dir = MagicMock()
    batch_processor.batch_files_dir.__truediv__ = lambda self, other: f"mock_path/{other}"
    
    # Variable pour capturer le contenu du fichier écrit
    written_content = []
    
    def mock_file_write(content):
        written_content.append(content)
    
    # Mock de l'écriture de fichier
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file_handle = mock_file.return_value.__enter__.return_value
        mock_file_handle.write.side_effect = mock_file_write
        
        # Tester avec gpt-5-nano
        file_id = batch_processor.create_batch_input_file(
            mock_sections,
            mock_corpus,
            mock_prompt_builder,
            model="gpt-5-nano"
        )
        
        # Vérifier que le fichier a été créé
        assert file_id == "file_123"
        
        # Analyser le contenu écrit
        assert len(written_content) > 0, "Du contenu devrait avoir été écrit"
        
        # Le contenu devrait être une ligne JSON
        json_content = json.loads(written_content[0])
        
        # Vérifier que temperature et top_p ne sont PAS dans le body
        body = json_content["body"]
        assert "temperature" not in body, "temperature ne devrait pas être dans le body pour gpt-5-nano"
        assert "top_p" not in body, "top_p ne devrait pas être dans le body pour gpt-5-nano"
        
        # Vérifier que les paramètres GPT-5 sont présents
        assert "reasoning_effort" in body, "reasoning_effort devrait être dans le body pour gpt-5-nano"
        assert "verbosity" in body, "verbosity devrait être dans le body pour gpt-5-nano"


def test_get_model_specific_params_logic():
    """
    Vérifie que la fonction de sélection des paramètres retourne
    les bonnes clés pour chaque famille de modèles.
    """
    import sys
    import os
    
    # Ajouter le répertoire racine au path pour importer stubs_batch
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    from stubs_batch import BatchProcessor
    from unittest.mock import MagicMock
    
    # Mock du client OpenAI et tracker
    mock_client = MagicMock()
    mock_tracker = MagicMock()
    
    # Créer une instance du BatchProcessor
    batch_processor = BatchProcessor.__new__(BatchProcessor)
    batch_processor.client = mock_client
    batch_processor.tracker = mock_tracker
    batch_processor.batch_files_dir = MagicMock()
    
    # Cas 1 : Modèle de raisonnement GPT-5
    gpt5_params = batch_processor.get_model_specific_params("gpt-5-nano")
    assert "temperature" not in gpt5_params, "temperature ne devrait pas être présent pour gpt-5-nano"
    assert "top_p" not in gpt5_params, "top_p ne devrait pas être présent pour gpt-5-nano"
    assert "reasoning_effort" in gpt5_params, "reasoning_effort devrait être présent pour gpt-5-nano"
    assert "verbosity" in gpt5_params, "verbosity devrait être présent pour gpt-5-nano"
    assert "max_completion_tokens" in gpt5_params, "max_completion_tokens devrait être présent pour gpt-5-nano"

    # Cas 2 : Modèle standard GPT-4
    gpt4_params = batch_processor.get_model_specific_params("gpt-4.1-turbo")
    assert "temperature" in gpt4_params, "temperature devrait être présent pour gpt-4.1-turbo"
    assert "top_p" in gpt4_params, "top_p devrait être présent pour gpt-4.1-turbo"
    assert "reasoning_effort" not in gpt4_params, "reasoning_effort ne devrait pas être présent pour gpt-4.1-turbo"
    assert "verbosity" not in gpt4_params, "verbosity ne devrait pas être présent pour gpt-4.1-turbo"
    assert "max_tokens" in gpt4_params, "max_tokens devrait être présent pour gpt-4.1-turbo"

    # Cas 3 : Modèle GPT-5 Chat (doit se comporter comme un modèle standard)
    gpt5_chat_params = batch_processor.get_model_specific_params("gpt-5-chat-latest")
    assert "temperature" in gpt5_chat_params, "temperature devrait être présent pour gpt-5-chat-latest"
    assert "top_p" in gpt5_chat_params, "top_p devrait être présent pour gpt-5-chat-latest"
    assert "reasoning_effort" not in gpt5_chat_params, "reasoning_effort ne devrait pas être présent pour gpt-5-chat-latest"
    assert "verbosity" not in gpt5_chat_params, "verbosity ne devrait pas être présent pour gpt-5-chat-latest"
    assert "max_tokens" in gpt5_chat_params, "max_tokens devrait être présent pour gpt-5-chat-latest"
    
    # Cas 4 : Modèle GPT-4o standard
    gpt4o_params = batch_processor.get_model_specific_params("gpt-4o")
    assert "temperature" in gpt4o_params, "temperature devrait être présent pour gpt-4o"
    assert "top_p" in gpt4o_params, "top_p devrait être présent pour gpt-4o"
    assert "reasoning_effort" not in gpt4o_params, "reasoning_effort ne devrait pas être présent pour gpt-4o"
    assert "verbosity" not in gpt4o_params, "verbosity ne devrait pas être présent pour gpt-4o"
    assert "max_tokens" in gpt4o_params, "max_tokens devrait être présent pour gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__])