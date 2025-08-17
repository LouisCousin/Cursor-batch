#!/usr/bin/env python3
"""
Tests pour l'intégration du mode Batch Anthropic.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import get_model_provider, get_model_details, ConfigManager
from src.core.process_tracker import ProcessTracker


class TestAnthropicIntegration:
    """Tests pour l'intégration Anthropic."""
    
    def test_model_provider_detection(self):
        """Test de la détection du fournisseur de modèle."""
        # Tests pour les modèles OpenAI
        assert get_model_provider("gpt-5") == "openai"
        assert get_model_provider("gpt-5-mini") == "openai"
        assert get_model_provider("gpt-4.1") == "openai"
        
        # Tests pour les modèles Anthropic
        assert get_model_provider("claude-sonnet-4-20250514") == "anthropic"
        assert get_model_provider("claude-3.5-sonnet-20240620") == "anthropic"
        
        # Test pour un modèle inconnu avec fallback
        assert get_model_provider("claude-unknown") == "anthropic"
        assert get_model_provider("gpt-unknown") == "openai"
    
    def test_model_details(self):
        """Test de la récupération des détails de modèle."""
        # Test modèle OpenAI
        details = get_model_details("gpt-5")
        assert details["provider"] == "openai"
        assert details["context"] == 400000
        assert details["max_output"] == 128000
        
        # Test modèle Anthropic
        details = get_model_details("claude-sonnet-4-20250514")
        assert details["provider"] == "anthropic"
        assert details["context"] == 200000
        assert details["max_output"] == 64000
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    def test_config_manager_api_key_retrieval(self):
        """Test de la récupération des clés API."""
        config_manager = ConfigManager()
        
        # Test récupération clé Anthropic depuis env
        api_key = config_manager.get_api_key("anthropic")
        assert api_key == "test-key"
        
        # Test erreur pour fournisseur non supporté
        with pytest.raises(ValueError, match="Fournisseur non supporté"):
            config_manager.get_api_key("unknown")
    
    def test_config_manager_validation(self):
        """Test de la validation de configuration de modèle."""
        config_manager = ConfigManager()
        
        with patch.object(config_manager, 'get_api_key', return_value='test-key'):
            # Test validation réussie
            details = config_manager.validate_model_config("claude-sonnet-4-20250514")
            assert details["provider"] == "anthropic"
        
        with patch.object(config_manager, 'get_api_key', side_effect=ValueError("Clé manquante")):
            # Test validation échouée
            with pytest.raises(ValueError, match="Configuration invalide"):
                config_manager.validate_model_config("claude-sonnet-4-20250514")
    
    def test_process_tracker_with_provider(self):
        """Test du tracker de processus avec support des fournisseurs."""
        tracker = ProcessTracker("data/test_process_db.json")
        
        plan_items = [
            {"code": "1.1", "title": "Test Section 1"},
            {"code": "1.2", "title": "Test Section 2"}
        ]
        
        # Test création processus Anthropic
        process_id = tracker.create_new_process(
            plan_items=plan_items,
            process_type="batch",
            description="Test Anthropic",
            provider="anthropic",
            model_name="claude-sonnet-4-20250514"
        )
        
        # Vérifier que le processus a été créé avec les bonnes informations
        process = tracker.get_process(process_id)
        assert process is not None
        assert process["provider"] == "anthropic"
        assert process["model_name"] == "claude-sonnet-4-20250514"
        
        # Test ajout batch Anthropic
        tracker.add_batch_to_process(
            process_id=process_id,
            batch_id="test-batch-123",
            section_codes=["1.1", "1.2"],
            batch_type="generation",
            provider="anthropic"
        )
        
        # Vérifier que le batch a été ajouté avec le bon fournisseur
        updated_process = tracker.get_process(process_id)
        batch_history = updated_process["batch_history"]
        assert len(batch_history) == 1
        assert batch_history[0]["provider"] == "anthropic"
        assert batch_history[0]["batch_id"] == "test-batch-123"
        
        # Test résumé du processus avec affichage du modèle
        summary = tracker.get_process_summary(process_id)
        assert summary is not None
        assert summary["provider"] == "anthropic"
        assert summary["model_name"] == "claude-sonnet-4-20250514"
        assert summary["model_display"] == "Anthropic: claude-sonnet-4-20250514"
        
        # Nettoyage
        tracker.delete_process(process_id)
    
    @patch('src.core.anthropic_batch_processor.anthropic')
    def test_anthropic_batch_processor_import(self, mock_anthropic):
        """Test de l'import et de l'initialisation du processeur Anthropic."""
        from src.core.anthropic_batch_processor import AnthropicBatchProcessor
        
        # Mock du client Anthropic
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Test création du processeur
        processor = AnthropicBatchProcessor("test-api-key")
        assert processor.api_key == "test-api-key"
        assert processor.client == mock_client
        
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test-api-key")
    
    @patch('src.core.anthropic_batch_processor.anthropic')
    def test_anthropic_batch_preparation(self, mock_anthropic):
        """Test de la préparation des requêtes batch Anthropic."""
        from src.core.anthropic_batch_processor import AnthropicBatchProcessor
        
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        processor = AnthropicBatchProcessor("test-api-key")
        
        # Test préparation des prompts
        prompts = [
            {
                "content": "Test prompt 1",
                "section_code": "1.1",
                "max_tokens": 2048,
                "temperature": 0.8
            },
            {
                "content": "Test prompt 2",
                "section_code": "1.2"
            }
        ]
        
        requests = processor.prepare_batch_requests(prompts, "claude-sonnet-4-20250514")
        
        assert len(requests) == 2
        
        # Vérifier la première requête
        req1 = requests[0]
        assert req1["params"]["model"] == "claude-sonnet-4-20250514"
        assert req1["params"]["messages"][0]["content"] == "Test prompt 1"
        assert req1["params"]["max_tokens"] == 2048
        assert req1["params"]["temperature"] == 0.8
        assert req1["metadata"]["section_code"] == "1.1"
        
        # Vérifier la deuxième requête avec valeurs par défaut
        req2 = requests[1]
        assert req2["params"]["max_tokens"] == 4096  # Valeur par défaut
        assert req2["params"]["temperature"] == 0.7  # Valeur par défaut
        assert req2["metadata"]["section_code"] == "1.2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
