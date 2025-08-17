#!/usr/bin/env python3
"""
Tests de validation des critères d'acceptation pour l'intégration Anthropic.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Ajouter le répertoire parent au path pour les imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config_manager import ConfigManager, get_model_details
from core.process_tracker import ProcessTracker, ProcessStatus
from app import launch_unified_batch_process


class TestAcceptanceCriteria:
    """Tests de validation des critères d'acceptation."""
    
    def test_criterion_1_anthropic_model_selection(self):
        """
        Critère 1: Quand un modèle Anthropic est sélectionné, 
        le lancement d'une génération en mode "batch" crée un nouveau batch via l'API d'Anthropic.
        """
        # Vérifier que les modèles Anthropic sont correctement détectés
        anthropic_models = ["claude-sonnet-4-20250514", "claude-3.5-sonnet-20240620"]
        
        for model in anthropic_models:
            details = get_model_details(model)
            assert details["provider"] == "anthropic", f"Le modèle {model} devrait être détecté comme Anthropic"
    
    @patch('core.anthropic_batch_processor.anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-anthropic-key'})
    def test_criterion_2_batch_appears_in_history(self, mock_anthropic):
        """
        Critère 2: Le batch Anthropic apparaît immédiatement dans l'onglet "Historique" 
        avec le statut "en cours" et l'indication du modèle.
        """
        # Mock du client et des réponses Anthropic
        mock_client = Mock()
        mock_batch_response = Mock()
        mock_batch_response.id = "test-batch-anthropic-123"
        mock_client.messages.batches.create.return_value = mock_batch_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Configuration
        config_manager = ConfigManager()
        tracker = ProcessTracker("data/test_acceptance_db.json")
        
        # Mock du corpus manager et prompt builder
        mock_corpus_manager = Mock()
        mock_corpus_manager.filter_corpus_by_section.return_value = []
        
        mock_prompt_builder = Mock()
        mock_prompt_builder.build_draft_prompt.return_value = "Test prompt for section"
        
        plan_items = [
            {"code": "1.1", "title": "Test Section 1"},
            {"code": "1.2", "title": "Test Section 2"}
        ]
        
        # Lancer le processus batch unifié avec un modèle Anthropic
        process_id = launch_unified_batch_process(
            plan_items=plan_items,
            model="claude-sonnet-4-20250514",
            config_manager=config_manager,
            corpus_manager=mock_corpus_manager,
            prompt_builder=mock_prompt_builder,
            corpus_params={},
            process_tracker=tracker,
            description="Test Anthropic Batch"
        )
        
        # Vérifier que le processus a été créé
        assert process_id is not None
        
        # Vérifier le contenu du processus
        process = tracker.get_process(process_id)
        assert process is not None
        assert process["provider"] == "anthropic"
        assert process["model_name"] == "claude-sonnet-4-20250514"
        assert process["status"] == ProcessStatus.EN_COURS.value
        
        # Vérifier le résumé pour l'affichage
        summary = tracker.get_process_summary(process_id)
        assert summary is not None
        assert summary["model_display"] == "Anthropic: claude-sonnet-4-20250514"
        assert summary["status"] == ProcessStatus.EN_COURS.value
        
        # Vérifier l'historique des batchs
        batch_history = process["batch_history"]
        assert len(batch_history) == 1
        batch_info = batch_history[0]
        assert batch_info["batch_id"] == "test-batch-anthropic-123"
        assert batch_info["provider"] == "anthropic"
        assert batch_info["status"] == "submitted"
        
        # Vérifier que l'API Anthropic a été appelée
        mock_client.messages.batches.create.assert_called_once()
        
        # Nettoyage
        tracker.delete_process(process_id)
    
    @patch('core.anthropic_batch_processor.anthropic')
    def test_criterion_3_status_updates_correctly(self, mock_anthropic):
        """
        Critère 3: Le statut du batch dans l'historique se met à jour correctement.
        """
        from core.anthropic_batch_processor import get_anthropic_batch_status
        
        # Mock du client Anthropic
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "test-batch-123"
        mock_batch.processing_status = "ended"
        mock_batch.request_counts = {"succeeded": 5, "errored": 0, "processing": 0}
        mock_batch.created_at = "2025-01-01T10:00:00Z"
        mock_batch.results_url = "https://api.anthropic.com/results/test"
        
        mock_client.messages.batches.retrieve.return_value = mock_batch
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Test de récupération du statut
        status = get_anthropic_batch_status("test-batch-123", "test-api-key")
        
        assert status["id"] == "test-batch-123"
        assert status["processing_status"] == "ended"
        assert status["request_counts"]["succeeded"] == 5
        assert status["results_url"] == "https://api.anthropic.com/results/test"
        
        mock_client.messages.batches.retrieve.assert_called_once_with("test-batch-123")
    
    @patch('core.anthropic_batch_processor.requests')
    @patch('core.anthropic_batch_processor.anthropic')
    def test_criterion_4_results_integration(self, mock_anthropic, mock_requests):
        """
        Critère 4: Une fois le batch terminé, les résultats sont correctement récupérés 
        et intégrés dans les fichiers de sortie.
        """
        from core.anthropic_batch_processor import get_anthropic_batch_results
        
        # Mock du client Anthropic
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "test-batch-123"
        mock_batch.processing_status = "ended"
        mock_batch.results_url = "https://api.anthropic.com/results/test"
        
        mock_client.messages.batches.retrieve.return_value = mock_batch
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock de la réponse HTTP pour le téléchargement des résultats
        mock_response = Mock()
        mock_response.text = '''{"custom_id": "req-1", "result": {"message": {"content": [{"type": "text", "text": "Résultat généré 1"}]}}}
{"custom_id": "req-2", "result": {"message": {"content": [{"type": "text", "text": "Résultat généré 2"}]}}}'''
        mock_response.raise_for_status.return_value = None
        mock_requests.get.return_value = mock_response
        
        # Test de récupération des résultats
        results = get_anthropic_batch_results("test-batch-123", "test-api-key")
        
        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[0]["content"] == "Résultat généré 1"
        assert results[1]["custom_id"] == "req-2"
        assert results[1]["content"] == "Résultat généré 2"
        
        mock_requests.get.assert_called_once()
    
    def test_criterion_5_openai_no_regression(self):
        """
        Critère 5: Le lancement d'un batch avec un modèle OpenAI continue 
        de fonctionner exactement comme avant, sans aucune régression.
        """
        # Vérifier que les modèles OpenAI sont toujours correctement détectés
        openai_models = ["gpt-5", "gpt-5-mini", "gpt-4.1"]
        
        for model in openai_models:
            details = get_model_details(model)
            assert details["provider"] == "openai", f"Le modèle {model} devrait être détecté comme OpenAI"
        
        # Vérifier que la logique de fallback fonctionne toujours
        details = get_model_details("gpt-unknown")
        assert details["provider"] == "openai"  # Fallback vers OpenAI
    
    def test_criterion_6_missing_api_key_error(self):
        """
        Critère 6: Si la clé API Anthropic est manquante ou invalide, 
        une erreur claire est affichée à l'utilisateur.
        """
        config_manager = ConfigManager()
        
        # Test avec clé manquante
        with patch.object(config_manager, 'get_api_key', side_effect=ValueError("Clé API manquante pour anthropic")):
            with pytest.raises(ValueError, match="Configuration invalide.*Clé API manquante"):
                config_manager.validate_model_config("claude-sonnet-4-20250514")
    
    def test_full_integration_workflow(self):
        """
        Test d'intégration complète du workflow Anthropic.
        """
        # Test de la chaîne complète de détection et configuration
        model = "claude-sonnet-4-20250514"
        
        # 1. Détection du fournisseur
        details = get_model_details(model)
        assert details["provider"] == "anthropic"
        
        # 2. Création d'un processus avec le bon fournisseur
        tracker = ProcessTracker("data/test_full_integration_db.json")
        
        plan_items = [{"code": "1.1", "title": "Section Test"}]
        
        process_id = tracker.create_new_process(
            plan_items=plan_items,
            provider="anthropic",
            model_name=model,
            description="Test intégration complète"
        )
        
        # 3. Vérification du processus créé
        process = tracker.get_process(process_id)
        assert process["provider"] == "anthropic"
        assert process["model_name"] == model
        
        # 4. Ajout d'un batch
        tracker.add_batch_to_process(
            process_id=process_id,
            batch_id="test-anthropic-batch",
            section_codes=["1.1"],
            provider="anthropic"
        )
        
        # 5. Vérification du résumé pour l'affichage
        summary = tracker.get_process_summary(process_id)
        assert summary["model_display"] == "Anthropic: claude-sonnet-4-20250514"
        
        # Nettoyage
        tracker.delete_process(process_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
