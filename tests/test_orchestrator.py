#!/usr/bin/env python3
"""
Tests unitaires pour le module orchestrateur.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from datetime import datetime

# Import des classes à tester
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.orchestrator import (
    GenerationTask, TaskStatus, OrchestrationContext,
    GenerationOrchestrator, create_linear_dependency_tasks
)


class TestGenerationTask:
    """Tests pour la classe GenerationTask."""
    
    def test_task_creation(self):
        """Test la création d'une tâche."""
        task = GenerationTask(
            id="task1",
            section_code="1.1",
            section_title="Introduction",
            dependencies=[]
        )
        
        assert task.id == "task1"
        assert task.section_code == "1.1"
        assert task.section_title == "Introduction"
        assert task.dependencies == []
        assert task.status == TaskStatus.PRET  # Pas de dépendances donc PRET
        assert task.result_text is None
        assert task.summary is None
        assert task.error_message is None
    
    def test_task_with_dependencies(self):
        """Test une tâche avec dépendances."""
        task = GenerationTask(
            id="task2",
            section_code="1.2",
            section_title="Contexte",
            dependencies=["task1"]
        )
        
        assert task.status == TaskStatus.EN_ATTENTE  # A des dépendances donc EN_ATTENTE
        assert task.dependencies == ["task1"]


class TestOrchestrationContext:
    """Tests pour la classe OrchestrationContext."""
    
    def test_context_creation(self):
        """Test la création d'un contexte."""
        context = OrchestrationContext()
        
        assert context.summaries == {}
        assert context.completed_tasks == {}
        assert isinstance(context.lock, threading.Lock)
    
    def test_add_summary_thread_safe(self):
        """Test l'ajout thread-safe de résumés."""
        context = OrchestrationContext()
        
        # Simuler des ajouts concurrents
        def add_summary(task_id, summary):
            context.add_summary(task_id, summary)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_summary, args=(f"task{i}", f"summary{i}"))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(context.summaries) == 10
        for i in range(10):
            assert f"task{i}" in context.summaries
            assert context.summaries[f"task{i}"] == f"summary{i}"
    
    def test_get_context_for_task(self):
        """Test la génération de contexte pour une tâche."""
        context = OrchestrationContext()
        context.add_summary("task1", "Résumé de la tâche 1")
        context.add_summary("task2", "Résumé de la tâche 2")
        
        # Tâche sans dépendances
        task_no_deps = GenerationTask("task3", "1.3", "Titre 3", [])
        context_text = context.get_context_for_task(task_no_deps)
        assert context_text == ""
        
        # Tâche avec dépendances
        task_with_deps = GenerationTask("task4", "1.4", "Titre 4", ["task1", "task2"])
        context_text = context.get_context_for_task(task_with_deps)
        
        assert "CONTEXTE DES SECTIONS PRÉCÉDENTES" in context_text
        assert "Résumé de la tâche 1" in context_text
        assert "Résumé de la tâche 2" in context_text


class TestGenerationOrchestrator:
    """Tests pour la classe GenerationOrchestrator."""
    
    def test_orchestrator_creation(self):
        """Test la création d'un orchestrateur."""
        tasks = [
            GenerationTask("task1", "1.1", "Titre 1", []),
            GenerationTask("task2", "1.2", "Titre 2", ["task1"])
        ]
        callback = Mock()
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        assert len(orchestrator.tasks) == 2
        assert "task1" in orchestrator.tasks
        assert "task2" in orchestrator.tasks
        assert orchestrator.progress_callback == callback
        assert orchestrator.max_workers == 2  # min(4, len(tasks))
    
    def test_get_ready_tasks(self):
        """Test l'identification des tâches prêtes."""
        tasks = [
            GenerationTask("task1", "1.1", "Titre 1", []),
            GenerationTask("task2", "1.2", "Titre 2", ["task1"]),
            GenerationTask("task3", "1.3", "Titre 3", ["task2"])
        ]
        callback = Mock()
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # Au début, seule task1 est prête (pas de dépendances)
        ready_tasks = orchestrator._get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task1"
        
        # Marquer task1 comme terminée
        orchestrator.tasks["task1"].status = TaskStatus.TERMINE
        
        # Maintenant task2 devrait être prête
        ready_tasks = orchestrator._get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task2"
    
    def test_extract_summary(self):
        """Test l'extraction de résumé."""
        tasks = [GenerationTask("task1", "1.1", "Titre 1", [])]
        callback = Mock()
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # Texte court
        short_text = "Ceci est un texte court."
        summary = orchestrator._extract_summary(short_text)
        assert summary == short_text
        
        # Texte long
        long_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4."
        summary = orchestrator._extract_summary(long_text)
        assert "Paragraph 1." in summary
        assert "Paragraph 4." in summary
        assert "[...]" in summary
    
    def test_update_dependent_tasks(self):
        """Test la mise à jour des tâches dépendantes."""
        tasks = [
            GenerationTask("task1", "1.1", "Titre 1", []),
            GenerationTask("task2", "1.2", "Titre 2", ["task1"]),
            GenerationTask("task3", "1.3", "Titre 3", ["task1", "task2"])
        ]
        callback = Mock()
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # task1 terminée
        completed_task = orchestrator.tasks["task1"]
        completed_task.status = TaskStatus.TERMINE
        
        orchestrator._update_dependent_tasks(completed_task)
        
        # task2 devrait maintenant être prête
        assert orchestrator.tasks["task2"].status == TaskStatus.PRET
        # task3 devrait encore être en attente (dépend aussi de task2)
        assert orchestrator.tasks["task3"].status == TaskStatus.EN_ATTENTE
    
    def test_execute_task_success(self):
        """Test l'exécution réussie d'une tâche."""
        task = GenerationTask("task1", "1.1", "Titre 1", [])
        callback = Mock()
        
        orchestrator = GenerationOrchestrator([task], callback)
        
        # Mock de la fonction de génération
        def mock_generation(task, context):
            return "Texte généré", "Résumé", True
        
        orchestrator.set_generation_function(mock_generation)
        
        result_task = orchestrator._execute_task(task)
        
        assert result_task.status == TaskStatus.TERMINE
        assert result_task.result_text == "Texte généré"
        assert result_task.summary == "Résumé"
        assert result_task.error_message is None
        assert result_task.start_time is not None
        assert result_task.end_time is not None
    
    def test_execute_task_failure(self):
        """Test l'exécution échouée d'une tâche."""
        task = GenerationTask("task1", "1.1", "Titre 1", [])
        callback = Mock()
        
        orchestrator = GenerationOrchestrator([task], callback)
        
        # Mock de la fonction de génération qui échoue
        def mock_generation(task, context):
            return None, "Erreur de génération", False
        
        orchestrator.set_generation_function(mock_generation)
        
        result_task = orchestrator._execute_task(task)
        
        assert result_task.status == TaskStatus.ECHEC
        assert result_task.result_text is None
        assert result_task.error_message == "La génération a échoué ou n'a pas produit de texte"
    
    def test_execute_task_exception(self):
        """Test l'exécution d'une tâche avec exception."""
        task = GenerationTask("task1", "1.1", "Titre 1", [])
        callback = Mock()
        
        orchestrator = GenerationOrchestrator([task], callback)
        
        # Mock de la fonction de génération qui lève une exception
        def mock_generation(task, context):
            raise ValueError("Erreur simulée")
        
        orchestrator.set_generation_function(mock_generation)
        
        result_task = orchestrator._execute_task(task)
        
        assert result_task.status == TaskStatus.ECHEC
        assert "Erreur simulée" in result_task.error_message
    
    def test_run_simple_workflow(self):
        """Test l'exécution d'un workflow simple."""
        tasks = [
            GenerationTask("task1", "1.1", "Titre 1", []),
            GenerationTask("task2", "1.2", "Titre 2", ["task1"])
        ]
        callback = Mock()
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # Mock de la fonction de génération
        def mock_generation(task, context):
            time.sleep(0.1)  # Simuler un traitement
            return f"Texte de {task.id}", f"Résumé de {task.id}", True
        
        orchestrator.set_generation_function(mock_generation)
        
        # Exécuter l'orchestrateur
        results = orchestrator.run()
        
        # Vérifier les résultats
        assert len(results) == 2
        assert results["task1"].status == TaskStatus.TERMINE
        assert results["task2"].status == TaskStatus.TERMINE
        assert results["task1"].result_text == "Texte de task1"
        assert results["task2"].result_text == "Texte de task2"
        
        # Vérifier que le callback a été appelé
        assert callback.call_count >= 2  # Au moins une fois par tâche
    
    def test_get_statistics(self):
        """Test le calcul des statistiques."""
        tasks = [
            GenerationTask("task1", "1.1", "Titre 1", []),
            GenerationTask("task2", "1.2", "Titre 2", ["task1"]),
            GenerationTask("task3", "1.3", "Titre 3", ["task2"])
        ]
        callback = Mock()
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # Marquer différents statuts
        orchestrator.tasks["task1"].status = TaskStatus.TERMINE
        orchestrator.tasks["task1"].start_time = datetime(2023, 1, 1, 10, 0, 0)
        orchestrator.tasks["task1"].end_time = datetime(2023, 1, 1, 10, 1, 0)
        
        orchestrator.tasks["task2"].status = TaskStatus.ECHEC
        orchestrator.tasks["task2"].start_time = datetime(2023, 1, 1, 10, 1, 0)
        orchestrator.tasks["task2"].end_time = datetime(2023, 1, 1, 10, 2, 0)
        
        orchestrator.tasks["task3"].status = TaskStatus.EN_COURS
        
        stats = orchestrator.get_statistics()
        
        assert stats['total_tasks'] == 3
        assert stats['completed'] == 1
        assert stats['failed'] == 1
        assert stats['in_progress'] == 1
        assert stats['waiting'] == 0
        assert stats['completion_rate'] == 100/3
        assert stats['total_execution_time'] == 120.0  # 2 minutes


class TestCreateLinearDependencyTasks:
    """Tests pour la fonction create_linear_dependency_tasks."""
    
    def test_create_tasks_no_dependencies(self):
        """Test avec une seule section."""
        sections = ["1.1 - Introduction"]
        tasks = create_linear_dependency_tasks(sections)
        
        assert len(tasks) == 1
        assert tasks[0].section_code == "1.1"
        assert tasks[0].section_title == "Introduction"
        assert tasks[0].dependencies == []
        assert tasks[0].status == TaskStatus.PRET
    
    def test_create_tasks_linear_dependencies(self):
        """Test avec plusieurs sections en dépendance linéaire."""
        sections = [
            "1.1 - Introduction",
            "1.2 - Contexte",
            "1.3 - Méthodologie"
        ]
        tasks = create_linear_dependency_tasks(sections)
        
        assert len(tasks) == 3
        
        # Première tâche : pas de dépendances
        assert tasks[0].section_code == "1.1"
        assert tasks[0].section_title == "Introduction"
        assert tasks[0].dependencies == []
        assert tasks[0].status == TaskStatus.PRET
        
        # Deuxième tâche : dépend de la première
        assert tasks[1].section_code == "1.2"
        assert tasks[1].section_title == "Contexte"
        assert tasks[1].dependencies == ["1.1_Introduction"]
        assert tasks[1].status == TaskStatus.EN_ATTENTE
        
        # Troisième tâche : dépend de la deuxième
        assert tasks[2].section_code == "1.3"
        assert tasks[2].section_title == "Méthodologie"
        assert tasks[2].dependencies == ["1.2_Contexte"]
        assert tasks[2].status == TaskStatus.EN_ATTENTE
    
    def test_create_tasks_without_separator(self):
        """Test avec des sections sans séparateur ' - '."""
        sections = ["Introduction", "Contexte"]
        tasks = create_linear_dependency_tasks(sections)
        
        assert len(tasks) == 2
        assert tasks[0].section_code == "SECTION_1"
        assert tasks[0].section_title == "Introduction"
        assert tasks[1].section_code == "SECTION_2"
        assert tasks[1].section_title == "Contexte"
        assert tasks[1].dependencies == ["SECTION_1_Introduction"]


# Tests d'intégration
class TestOrchestrationIntegration:
    """Tests d'intégration pour l'orchestrateur."""
    
    def test_full_workflow_simulation(self):
        """Test d'un workflow complet simulé."""
        sections = [
            "1.1 - Introduction",
            "1.2 - État de l'art",
            "1.3 - Méthodologie",
            "1.4 - Résultats"
        ]
        
        tasks = create_linear_dependency_tasks(sections)
        callback_calls = []
        
        def callback(tasks_list):
            callback_calls.append([task.status.value for task in tasks_list])
        
        orchestrator = GenerationOrchestrator(tasks, callback)
        
        # Simuler une fonction de génération réaliste
        def mock_generation(task, context):
            # Simuler un délai de traitement
            time.sleep(0.05)
            
            # Vérifier que le contexte est fourni pour les tâches dépendantes
            if task.dependencies and not context:
                return None, "Contexte manquant", False
            
            text = f"Contenu généré pour {task.section_title}"
            if context:
                text = f"Basé sur le contexte précédent. {text}"
            
            summary = f"Résumé de {task.section_title}"
            return text, summary, True
        
        orchestrator.set_generation_function(mock_generation)
        
        # Exécuter l'orchestrateur
        results = orchestrator.run()
        
        # Vérifications
        assert len(results) == 4
        for task_id, task in results.items():
            assert task.status == TaskStatus.TERMINE
            assert task.result_text is not None
            assert task.summary is not None
            assert task.start_time is not None
            assert task.end_time is not None
        
        # Vérifier que les tâches dépendantes ont du contexte
        task2_result = results["1.2_État de l'art"]
        assert "Basé sur le contexte précédent" in task2_result.result_text
        
        # Vérifier que le callback a été appelé plusieurs fois
        assert len(callback_calls) > 4
        
        # Vérifier la progression logique des statuts
        final_call = callback_calls[-1]
        assert all(status == "TERMINÉ" for status in final_call)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
