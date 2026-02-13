#!/usr/bin/env python3
"""
Module d'orchestration pour la génération automatique d'ouvrages.
Gère l'exécution en parallèle et les dépendances entre les sections.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum


class TaskStatus(Enum):
    """Statuts possibles d'une tâche de génération."""
    EN_ATTENTE = "EN_ATTENTE"
    PRET = "PRÊT"
    EN_COURS = "EN_COURS"
    TERMINE = "TERMINÉ"
    ECHEC = "ÉCHEC"


@dataclass
class GenerationTask:
    """
    Représente une tâche de génération de section.
    """
    id: str
    section_code: str
    section_title: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.EN_ATTENTE
    result_text: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialise le statut selon les dépendances."""
        if not self.dependencies:
            self.status = TaskStatus.PRET


@dataclass
class OrchestrationContext:
    """
    Contexte partagé entre les tâches contenant les résumés et états.
    """
    summaries: Dict[str, str] = field(default_factory=dict)
    completed_tasks: Dict[str, GenerationTask] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_summary(self, task_id: str, summary: str) -> None:
        """Ajoute un résumé de manière thread-safe."""
        with self.lock:
            self.summaries[task_id] = summary
    
    def add_completed_task(self, task: GenerationTask) -> None:
        """Ajoute une tâche terminée de manière thread-safe."""
        with self.lock:
            self.completed_tasks[task.id] = task
    
    def get_context_for_task(self, task: GenerationTask) -> str:
        """
        Génère le contexte textuel pour une tâche basé sur les résumés précédents.
        """
        with self.lock:
            if not task.dependencies:
                return ""
            
            context_parts = []
            for dep_id in task.dependencies:
                if dep_id in self.summaries:
                    summary = self.summaries[dep_id]
                    context_parts.append(f"Résumé de la section précédente ({dep_id}): {summary}")
            
            if context_parts:
                return "\n\n--- CONTEXTE DES SECTIONS PRÉCÉDENTES ---\n" + "\n\n".join(context_parts) + "\n--- FIN DU CONTEXTE ---\n\n"
            return ""


class GenerationOrchestrator:
    """
    Orchestrateur principal pour la génération automatique d'ouvrages.
    Gère l'exécution en parallèle des tâches avec dépendances.
    """
    
    def __init__(self, tasks: List[GenerationTask], progress_callback: Callable[[List[GenerationTask]], None]):
        """
        Initialise l'orchestrateur.
        
        Args:
            tasks: Liste des tâches à exécuter
            progress_callback: Fonction appelée à chaque changement de statut
        """
        self.tasks = {task.id: task for task in tasks}
        self.progress_callback = progress_callback
        self.context = OrchestrationContext()
        self.max_workers = max(1, min(4, len(tasks)))  # Minimum 1 pour éviter ValueError
        self._generation_function = None
        self._should_stop = False
        self._tasks_lock = threading.Lock()
    
    def set_generation_function(self, func: Callable[[GenerationTask, str], tuple]) -> None:
        """
        Définit la fonction de génération à utiliser.
        
        Args:
            func: Fonction qui prend (task, context) et retourne (text, summary, success)
        """
        self._generation_function = func
    
    def stop(self) -> None:
        """Demande l'arrêt de l'orchestrateur."""
        self._should_stop = True
    
    def _extract_summary(self, text: str) -> str:
        """
        Extrait un résumé du texte généré.
        Utilise les premiers et derniers paragraphes pour créer un résumé.
        """
        if not text:
            return ""
        
        # Divise le texte en paragraphes
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 2:
            return text[:500] + "..." if len(text) > 500 else text
        
        # Prend le premier et dernier paragraphe
        summary_parts = [paragraphs[0]]
        if len(paragraphs) > 1:
            summary_parts.append(paragraphs[-1])
        
        summary = " [...] ".join(summary_parts)
        return summary[:800] + "..." if len(summary) > 800 else summary
    
    def _get_ready_tasks(self) -> List[GenerationTask]:
        """Retourne les tâches prêtes à être exécutées."""
        ready_tasks = []

        with self._tasks_lock:
            for task in self.tasks.values():
                if task.status == TaskStatus.PRET:
                    ready_tasks.append(task)
                elif task.status == TaskStatus.EN_ATTENTE:
                    # Vérifier si toutes les dépendances sont terminées
                    # Note: les dépendances référençant des IDs inexistants ne sont pas
                    # considérées comme terminées pour éviter les lancements prématurés
                    existing_deps = [dep_id for dep_id in task.dependencies if dep_id in self.tasks]
                    if len(existing_deps) != len(task.dependencies):
                        # Certaines dépendances référencent des tâches inexistantes
                        continue
                    dependencies_completed = all(
                        self.tasks[dep_id].status == TaskStatus.TERMINE
                        for dep_id in existing_deps
                    )
                    if dependencies_completed:
                        task.status = TaskStatus.PRET
                        ready_tasks.append(task)

        return ready_tasks

    def _update_dependent_tasks(self, completed_task: GenerationTask) -> None:
        """Met à jour le statut des tâches dépendantes après completion d'une tâche."""
        with self._tasks_lock:
            for task in self.tasks.values():
                if (task.status == TaskStatus.EN_ATTENTE and
                    completed_task.id in task.dependencies):

                    # Vérifier si toutes les dépendances sont maintenant terminées
                    existing_deps = [dep_id for dep_id in task.dependencies if dep_id in self.tasks]
                    if len(existing_deps) != len(task.dependencies):
                        continue
                    dependencies_completed = all(
                        self.tasks[dep_id].status == TaskStatus.TERMINE
                        for dep_id in existing_deps
                    )
                    if dependencies_completed:
                        task.status = TaskStatus.PRET
    
    def _execute_task(self, task: GenerationTask) -> GenerationTask:
        """
        Exécute une tâche de génération.
        
        Args:
            task: Tâche à exécuter
            
        Returns:
            La tâche mise à jour avec le résultat
        """
        if self._should_stop:
            task.status = TaskStatus.ECHEC
            task.error_message = "Arrêt demandé par l'utilisateur"
            return task
        
        if not self._generation_function:
            task.status = TaskStatus.ECHEC
            task.error_message = "Fonction de génération non définie"
            return task
        
        try:
            task.status = TaskStatus.EN_COURS
            task.start_time = datetime.now()
            
            # Construire le contexte pour cette tâche
            context = self.context.get_context_for_task(task)
            
            # Exécuter la génération
            text, summary, success = self._generation_function(task, context)
            
            task.end_time = datetime.now()
            
            if success and text:
                task.result_text = text
                task.summary = summary or self._extract_summary(text)
                task.status = TaskStatus.TERMINE
                
                # Ajouter au contexte partagé
                self.context.add_summary(task.id, task.summary)
                self.context.add_completed_task(task)
            else:
                task.status = TaskStatus.ECHEC
                task.error_message = "La génération a échoué ou n'a pas produit de texte"
        
        except Exception as e:
            task.status = TaskStatus.ECHEC
            task.error_message = f"Erreur lors de la génération: {str(e)}"
            task.end_time = datetime.now()
        
        return task
    
    def run(self) -> Dict[str, GenerationTask]:
        """
        Lance l'exécution de toutes les tâches avec gestion des dépendances.
        Version synchrone pour compatibilité Streamlit.
        
        Returns:
            Dictionnaire des tâches avec leur résultat final
        """
        if not self._generation_function:
            raise ValueError("Fonction de génération non définie. Utilisez set_generation_function().")
        
        # Callback initial
        self.progress_callback(list(self.tasks.values()))
        
        # Boucle principale d'exécution synchrone
        while not self._should_stop:
            # Identifier les tâches prêtes
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # Vérifier si toutes les tâches sont terminées ou en échec
                remaining_tasks = [
                    task for task in self.tasks.values() 
                    if task.status not in [TaskStatus.TERMINE, TaskStatus.ECHEC]
                ]
                
                if not remaining_tasks:
                    break  # Toutes les tâches sont terminées
                
                # Attendre un peu avant de revérifier
                time.sleep(0.1)
                continue
            
            # Traiter les tâches prêtes une par une (compatible Streamlit)
            for task in ready_tasks:
                if self._should_stop:
                    break
                
                if task.status == TaskStatus.PRET:
                    try:
                        # Exécuter la tâche de manière synchrone
                        completed_task = self._execute_task(task)
                        
                        # Mettre à jour la tâche dans notre dictionnaire
                        self.tasks[completed_task.id] = completed_task
                        
                        # Mettre à jour les tâches dépendantes
                        if completed_task.status == TaskStatus.TERMINE:
                            self._update_dependent_tasks(completed_task)
                        
                        # Notifier l'interface
                        self.progress_callback(list(self.tasks.values()))
                        
                    except Exception as e:
                        # En cas d'erreur non capturée
                        task.status = TaskStatus.ECHEC
                        task.error_message = f"Erreur inattendue: {str(e)}"
                        task.end_time = datetime.now()
                        self.tasks[task.id] = task
                        self.progress_callback(list(self.tasks.values()))
        
        return self.tasks
    
    def run_parallel(self) -> Dict[str, GenerationTask]:
        """
        Lance l'exécution en parallèle (peut causer des warnings Streamlit).
        Utiliser uniquement en dehors de Streamlit.
        
        Returns:
            Dictionnaire des tâches avec leur résultat final
        """
        if not self._generation_function:
            raise ValueError("Fonction de génération non définie. Utilisez set_generation_function().")
        
        # Callback initial
        self.progress_callback(list(self.tasks.values()))
        
        # Boucle principale d'exécution parallèle
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self._should_stop:
                # Identifier les tâches prêtes
                ready_tasks = self._get_ready_tasks()
                
                if not ready_tasks:
                    # Vérifier si toutes les tâches sont terminées ou en échec
                    remaining_tasks = [
                        task for task in self.tasks.values() 
                        if task.status not in [TaskStatus.TERMINE, TaskStatus.ECHEC]
                    ]
                    
                    if not remaining_tasks:
                        break  # Toutes les tâches sont terminées
                    
                    # Attendre un peu avant de revérifier
                    time.sleep(0.1)
                    continue
                
                # Soumettre les tâches prêtes à l'exécuteur
                future_to_task = {}
                for task in ready_tasks:
                    if task.status == TaskStatus.PRET:
                        future = executor.submit(self._execute_task, task)
                        future_to_task[future] = task
                
                # Attendre la completion des tâches
                for future in as_completed(future_to_task):
                    if self._should_stop:
                        break

                    task = future_to_task[future]
                    try:
                        completed_task = future.result()

                        # Mettre à jour la tâche dans notre dictionnaire (protégé par lock)
                        with self._tasks_lock:
                            self.tasks[completed_task.id] = completed_task

                        # Mettre à jour les tâches dépendantes
                        if completed_task.status == TaskStatus.TERMINE:
                            self._update_dependent_tasks(completed_task)

                        # Notifier l'interface
                        self.progress_callback(list(self.tasks.values()))

                    except Exception as e:
                        # En cas d'erreur non capturée
                        with self._tasks_lock:
                            task.status = TaskStatus.ECHEC
                            task.error_message = f"Erreur inattendue: {str(e)}"
                            task.end_time = datetime.now()
                            self.tasks[task.id] = task
                        self.progress_callback(list(self.tasks.values()))
        
        return self.tasks
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'exécution.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        total_tasks = len(self.tasks)
        completed = len([t for t in self.tasks.values() if t.status == TaskStatus.TERMINE])
        failed = len([t for t in self.tasks.values() if t.status == TaskStatus.ECHEC])
        in_progress = len([t for t in self.tasks.values() if t.status == TaskStatus.EN_COURS])
        waiting = len([t for t in self.tasks.values() if t.status in [TaskStatus.EN_ATTENTE, TaskStatus.PRET]])
        
        # Calculer le temps total d'exécution
        start_times = [t.start_time for t in self.tasks.values() if t.start_time]
        end_times = [t.end_time for t in self.tasks.values() if t.end_time]
        
        total_time = None
        if start_times and end_times:
            earliest_start = min(start_times)
            latest_end = max(end_times)
            total_time = (latest_end - earliest_start).total_seconds()
        
        return {
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'waiting': waiting,
            'completion_rate': (completed / total_tasks * 100) if total_tasks > 0 else 0,
            'total_execution_time': total_time
        }


def create_linear_dependency_tasks(sections: List[str]) -> List[GenerationTask]:
    """
    Crée une liste de tâches avec des dépendances linéaires.
    
    Args:
        sections: Liste des sections au format "CODE - TITRE"
        
    Returns:
        Liste des tâches avec dépendances linéaires
    """
    tasks = []
    
    for i, section_full in enumerate(sections):
        # Extraire le code et le titre
        if " - " in section_full:
            section_code, section_title = section_full.split(" - ", 1)
        else:
            section_code, section_title = f"SECTION_{i+1}", section_full
        
        # Créer l'ID de la tâche
        task_id = f"{section_code}_{section_title}"
        
        # Définir les dépendances (chaque tâche dépend de la précédente)
        dependencies = []
        if i > 0:
            # Dépend de la tâche précédente
            prev_section = sections[i-1]
            if " - " in prev_section:
                prev_code, prev_title = prev_section.split(" - ", 1)
            else:
                prev_code, prev_title = f"SECTION_{i}", prev_section
            dependencies = [f"{prev_code}_{prev_title}"]
        
        # Créer la tâche
        task = GenerationTask(
            id=task_id,
            section_code=section_code,
            section_title=section_title,
            dependencies=dependencies
        )
        
        tasks.append(task)
    
    return tasks
