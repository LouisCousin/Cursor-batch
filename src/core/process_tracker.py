#!/usr/bin/env python3
"""
Module de suivi des processus de génération.
Gère l'état persistant des processus de génération par lot et permet la reprise.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import threading
import logging

try:
    from tinydb import TinyDB, Query
except ImportError:
    # Fallback vers JSON si TinyDB n'est pas disponible
    TinyDB = None
    Query = None


class ProcessStatus(Enum):
    """Statuts possibles d'un processus de génération."""
    EN_ATTENTE = "en_attente"
    EN_COURS = "en_cours"
    TERMINE = "terminé"
    EN_ECHEC = "en_echec"
    ANNULE = "annulé"


class SectionStatus(Enum):
    """Statuts possibles d'une section dans un processus."""
    EN_ATTENTE = "en_attente"
    EN_COURS = "en_cours"
    SUCCES = "succes"
    ECHEC = "echec"


class ProcessTracker:
    """
    Gestionnaire persistant des processus de génération.
    Utilise TinyDB pour la persistance ou un fallback JSON.
    """
    
    def __init__(self, db_path: str = "data/process_db.json"):
        """
        Initialise le tracker avec la base de données.
        
        Args:
            db_path: Chemin vers le fichier de base de données
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Initialiser la base de données
        if TinyDB is not None:
            self.db = TinyDB(str(self.db_path))
            self.processes_table = self.db.table('processes')
            self._use_tinydb = True
        else:
            # Fallback vers JSON
            self._use_tinydb = False
            self._load_json_db()
            
        logging.info(f"ProcessTracker initialisé avec {'TinyDB' if self._use_tinydb else 'JSON'}")
    
    def _load_json_db(self):
        """Charge la base de données JSON si TinyDB n'est pas disponible."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self._json_data = json.load(f)
            else:
                self._json_data = {"processes": []}
        except Exception as e:
            logging.warning(f"Erreur lors du chargement de la DB JSON: {e}")
            self._json_data = {"processes": []}
    
    def _save_json_db(self):
        """Sauvegarde la base de données JSON."""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self._json_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de la DB JSON: {e}")
    
    def create_new_process(self, plan_items: List[Dict[str, Any]], 
                          process_type: str = "batch", 
                          description: str = "",
                          provider: str = "openai",
                          model_name: str = "") -> str:
        """
        Crée un nouveau processus de génération.
        
        Args:
            plan_items: Liste des sections à traiter
            process_type: Type de processus ("batch" ou "synchrone")
            description: Description optionnelle du processus
            provider: Fournisseur de l'API ("openai" ou "anthropic")
            model_name: Nom du modèle utilisé
            
        Returns:
            ID unique du processus créé
        """
        with self._lock:
            process_id = str(uuid.uuid4())
            
            # Créer les sections avec statut initial
            sections = []
            for item in plan_items:
                section = {
                    "section_code": item.get('code', ''),
                    "section_title": item.get('title', ''),
                    "status": SectionStatus.EN_ATTENTE.value,
                    "batch_id": None,
                    "error_message": None,
                    "start_time": None,
                    "end_time": None,
                    "result_path": None
                }
                sections.append(section)
            
            # Créer le processus
            process_data = {
                "process_id": process_id,
                "type": process_type,
                "description": description,
                "provider": provider,
                "model_name": model_name,
                "status": ProcessStatus.EN_ATTENTE.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total_sections": len(sections),
                "completed_sections": 0,
                "failed_sections": 0,
                "sections": sections,
                "batch_history": []  # Historique des batchs lancés
            }
            
            # Sauvegarder selon le type de DB
            if self._use_tinydb:
                self.processes_table.insert(process_data)
            else:
                self._json_data["processes"].append(process_data)
                self._save_json_db()
            
            logging.info(f"Nouveau processus créé: {process_id} ({len(sections)} sections)")
            return process_id
    
    def update_process_status(self, process_id: str, status: ProcessStatus):
        """Met à jour le statut global d'un processus."""
        with self._lock:
            if self._use_tinydb:
                Process = Query()
                self.processes_table.update({
                    'status': status.value,
                    'updated_at': datetime.now().isoformat()
                }, Process.process_id == process_id)
            else:
                for process in self._json_data["processes"]:
                    if process["process_id"] == process_id:
                        process["status"] = status.value
                        process["updated_at"] = datetime.now().isoformat()
                        break
                self._save_json_db()
    
    def update_section_status(self, process_id: str, section_code: str, 
                             status: SectionStatus, batch_id: str = None,
                             error_message: str = None, result_path: str = None):
        """
        Met à jour le statut d'une section spécifique.
        
        Args:
            process_id: ID du processus
            section_code: Code de la section
            status: Nouveau statut
            batch_id: ID du batch OpenAI associé
            error_message: Message d'erreur le cas échéant
            result_path: Chemin vers le fichier de résultat
        """
        with self._lock:
            if self._use_tinydb:
                Process = Query()
                processes = self.processes_table.search(Process.process_id == process_id)
                if processes:
                    process = processes[0]
                    # Mettre à jour la section
                    for section in process['sections']:
                        if section['section_code'] == section_code:
                            section['status'] = status.value
                            section['batch_id'] = batch_id
                            section['error_message'] = error_message
                            section['result_path'] = result_path
                            
                            if status == SectionStatus.EN_COURS:
                                section['start_time'] = datetime.now().isoformat()
                            elif status in [SectionStatus.SUCCES, SectionStatus.ECHEC]:
                                section['end_time'] = datetime.now().isoformat()
                            break
                    
                    # Recalculer les compteurs
                    completed = len([s for s in process['sections'] if s['status'] == SectionStatus.SUCCES.value])
                    failed = len([s for s in process['sections'] if s['status'] == SectionStatus.ECHEC.value])
                    
                    # Mettre à jour le processus
                    self.processes_table.update({
                        'sections': process['sections'],
                        'completed_sections': completed,
                        'failed_sections': failed,
                        'updated_at': datetime.now().isoformat()
                    }, Process.process_id == process_id)
                    
                    # Mettre à jour le statut global si nécessaire
                    if completed + failed == process['total_sections']:
                        global_status = ProcessStatus.TERMINE if failed == 0 else ProcessStatus.EN_ECHEC
                        self.update_process_status(process_id, global_status)
            else:
                # Version JSON
                for process in self._json_data["processes"]:
                    if process["process_id"] == process_id:
                        # Mettre à jour la section
                        for section in process['sections']:
                            if section['section_code'] == section_code:
                                section['status'] = status.value
                                section['batch_id'] = batch_id
                                section['error_message'] = error_message
                                section['result_path'] = result_path
                                
                                if status == SectionStatus.EN_COURS:
                                    section['start_time'] = datetime.now().isoformat()
                                elif status in [SectionStatus.SUCCES, SectionStatus.ECHEC]:
                                    section['end_time'] = datetime.now().isoformat()
                                break
                        
                        # Recalculer les compteurs
                        completed = len([s for s in process['sections'] if s['status'] == SectionStatus.SUCCES.value])
                        failed = len([s for s in process['sections'] if s['status'] == SectionStatus.ECHEC.value])
                        
                        process['completed_sections'] = completed
                        process['failed_sections'] = failed
                        process['updated_at'] = datetime.now().isoformat()
                        
                        # Mettre à jour le statut global si nécessaire
                        if completed + failed == process['total_sections']:
                            process['status'] = ProcessStatus.TERMINE.value if failed == 0 else ProcessStatus.EN_ECHEC.value
                        
                        break
                
                self._save_json_db()
    
    def add_batch_to_process(self, process_id: str, batch_id: str, 
                           section_codes: List[str], batch_type: str = "generation",
                           provider: str = "openai"):
        """
        Ajoute un batch à l'historique d'un processus.
        
        Args:
            process_id: ID du processus
            batch_id: ID du batch (OpenAI ou Anthropic)
            section_codes: Liste des codes de sections traitées par ce batch
            batch_type: Type de batch ("generation", "resume", etc.)
            provider: Fournisseur du batch ("openai" ou "anthropic")
        """
        with self._lock:
            batch_info = {
                "batch_id": batch_id,
                "batch_type": batch_type,
                "provider": provider,
                "section_codes": section_codes,
                "created_at": datetime.now().isoformat(),
                "status": "submitted"
            }
            
            if self._use_tinydb:
                Process = Query()
                processes = self.processes_table.search(Process.process_id == process_id)
                if processes:
                    process = processes[0]
                    process['batch_history'].append(batch_info)
                    self.processes_table.update({
                        'batch_history': process['batch_history'],
                        'updated_at': datetime.now().isoformat()
                    }, Process.process_id == process_id)
            else:
                for process in self._json_data["processes"]:
                    if process["process_id"] == process_id:
                        process['batch_history'].append(batch_info)
                        process['updated_at'] = datetime.now().isoformat()
                        break
                self._save_json_db()
    
    def get_failed_or_pending_sections(self, process_id: str) -> List[Dict[str, Any]]:
        """
        Retourne la liste des sections qui ne sont pas en statut 'succes'.
        
        Args:
            process_id: ID du processus
            
        Returns:
            Liste des sections à reprendre
        """
        process = self.get_process(process_id)
        if not process:
            return []
        
        failed_sections = []
        for section in process['sections']:
            if section['status'] in [SectionStatus.EN_ATTENTE.value, SectionStatus.ECHEC.value]:
                failed_sections.append({
                    'code': section['section_code'],
                    'title': section['section_title'],
                    'status': section['status'],
                    'error_message': section.get('error_message')
                })
        
        return failed_sections
    
    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un processus par son ID.
        
        Args:
            process_id: ID du processus
            
        Returns:
            Données du processus ou None si non trouvé
        """
        if self._use_tinydb:
            Process = Query()
            processes = self.processes_table.search(Process.process_id == process_id)
            return processes[0] if processes else None
        else:
            for process in self._json_data["processes"]:
                if process["process_id"] == process_id:
                    return process
            return None
    
    def get_all_processes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retourne l'historique de tous les processus.
        
        Args:
            limit: Nombre maximum de processus à retourner
            
        Returns:
            Liste des processus triés par date de création (plus récent en premier)
        """
        if self._use_tinydb:
            all_processes = self.processes_table.all()
        else:
            all_processes = self._json_data["processes"].copy()
        
        # Trier par date de création (plus récent en premier)
        all_processes.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Limiter le nombre de résultats
        return all_processes[:limit]
    
    def get_process_summary(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne un résumé d'un processus pour l'affichage.
        
        Args:
            process_id: ID du processus
            
        Returns:
            Résumé du processus
        """
        process = self.get_process(process_id)
        if not process:
            return None
        
        # Calculer les statistiques
        total = process['total_sections']
        completed = process['completed_sections']
        failed = process['failed_sections']
        in_progress = len([s for s in process['sections'] if s['status'] == SectionStatus.EN_COURS.value])
        pending = total - completed - failed - in_progress
        
        # Créer le résumé de statut
        status = process['status']
        if status == ProcessStatus.EN_COURS.value:
            status_text = f"En cours ({completed}/{total} terminées)"
        elif status == ProcessStatus.EN_ECHEC.value:
            status_text = f"En échec ({completed}/{total} terminées, {failed} échecs)"
        elif status == ProcessStatus.TERMINE.value:
            status_text = f"Terminé ({completed}/{total} sections)"
        else:
            status_text = status.replace('_', ' ').title()
        
        # Formater l'affichage du modèle avec le fournisseur
        provider = process.get('provider', 'openai')
        model_name = process.get('model_name', '')
        
        if provider and model_name:
            model_display = f"{provider.title()}: {model_name}"
        elif model_name:
            model_display = model_name
        else:
            model_display = "Non spécifié"
        
        return {
            "process_id": process_id,
            "type": process.get('type', 'batch'),
            "description": process.get('description', ''),
            "provider": provider,
            "model_name": model_name,
            "model_display": model_display,
            "status": status,
            "status_text": status_text,
            "created_at": process['created_at'],
            "updated_at": process['updated_at'],
            "total_sections": total,
            "completed_sections": completed,
            "failed_sections": failed,
            "in_progress_sections": in_progress,
            "pending_sections": pending,
            "can_resume": status == ProcessStatus.EN_ECHEC.value and (failed > 0 or pending > 0),
            "completion_rate": (completed / total * 100) if total > 0 else 0
        }
    
    def delete_process(self, process_id: str) -> bool:
        """
        Supprime un processus de l'historique.
        
        Args:
            process_id: ID du processus à supprimer
            
        Returns:
            True si supprimé avec succès, False sinon
        """
        with self._lock:
            if self._use_tinydb:
                Process = Query()
                result = self.processes_table.remove(Process.process_id == process_id)
                return len(result) > 0
            else:
                original_length = len(self._json_data["processes"])
                self._json_data["processes"] = [
                    p for p in self._json_data["processes"] 
                    if p["process_id"] != process_id
                ]
                if len(self._json_data["processes"]) < original_length:
                    self._save_json_db()
                    return True
                return False
    
    def cleanup_old_processes(self, days_old: int = 30) -> int:
        """
        Nettoie les anciens processus terminés.
        
        Args:
            days_old: Nombre de jours d'ancienneté pour le nettoyage
            
        Returns:
            Nombre de processus supprimés
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_iso = cutoff_date.isoformat()
        
        deleted_count = 0
        
        with self._lock:
            if self._use_tinydb:
                Process = Query()
                to_delete = self.processes_table.search(
                    (Process.status.one_of([ProcessStatus.TERMINE.value, ProcessStatus.ANNULE.value])) &
                    (Process.created_at < cutoff_iso)
                )
                for process in to_delete:
                    self.processes_table.remove(Process.process_id == process['process_id'])
                    deleted_count += 1
            else:
                original_processes = self._json_data["processes"].copy()
                self._json_data["processes"] = [
                    p for p in self._json_data["processes"]
                    if not (p['status'] in [ProcessStatus.TERMINE.value, ProcessStatus.ANNULE.value] and
                           p['created_at'] < cutoff_iso)
                ]
                deleted_count = len(original_processes) - len(self._json_data["processes"])
                if deleted_count > 0:
                    self._save_json_db()
        
        if deleted_count > 0:
            logging.info(f"Nettoyage: {deleted_count} anciens processus supprimés")
        
        return deleted_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques globales sur tous les processus.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        all_processes = self.get_all_processes()
        
        stats = {
            "total_processes": len(all_processes),
            "completed_processes": 0,
            "failed_processes": 0,
            "in_progress_processes": 0,
            "total_sections": 0,
            "completed_sections": 0,
            "failed_sections": 0,
            "average_completion_rate": 0
        }
        
        if not all_processes:
            return stats
        
        total_completion_rate = 0
        
        for process in all_processes:
            status = process['status']
            if status == ProcessStatus.TERMINE.value:
                stats["completed_processes"] += 1
            elif status == ProcessStatus.EN_ECHEC.value:
                stats["failed_processes"] += 1
            elif status == ProcessStatus.EN_COURS.value:
                stats["in_progress_processes"] += 1
            
            stats["total_sections"] += process['total_sections']
            stats["completed_sections"] += process['completed_sections']
            stats["failed_sections"] += process['failed_sections']
            
            completion_rate = (process['completed_sections'] / process['total_sections'] * 100) if process['total_sections'] > 0 else 0
            total_completion_rate += completion_rate
        
        stats["average_completion_rate"] = total_completion_rate / len(all_processes)
        
        return stats
