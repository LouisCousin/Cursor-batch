#!/usr/bin/env python3
"""
Module de gestion des traitements par lot via l'API Batch d'OpenAI.
Gère la création, le suivi et la reprise des processus de génération par lot.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    logging.warning("OpenAI non disponible. Fonctionnalités de batch désactivées.")

# Import des modules locaux
import sys
sys.path.append('src')

from core.process_tracker import ProcessTracker, ProcessStatus, SectionStatus
from core.prompt_builder import PromptBuilder
from core.corpus_manager import CorpusManager


class BatchProcessor:
    """
    Gestionnaire des traitements par lot via l'API Batch d'OpenAI.
    """
    
    def __init__(self, api_key: str, process_tracker: ProcessTracker = None):
        """
        Initialise le processeur de batch.
        
        Args:
            api_key: Clé API OpenAI
            process_tracker: Tracker de processus (optionnel)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI non disponible. Impossible d'utiliser le traitement par lot.")
        
        self.client = OpenAI(api_key=api_key)
        self.tracker = process_tracker or ProcessTracker()
        self.batch_files_dir = Path("data/batch_files")
        self.batch_files_dir.mkdir(parents=True, exist_ok=True)
        
    def create_batch_input_file(self, sections_data: List[Dict[str, Any]], 
                              corpus_manager: CorpusManager,
                              prompt_builder: PromptBuilder,
                              model: str = "gpt-4o-mini",
                              corpus_params: Dict[str, Any] = None) -> str:
        """
        Crée le fichier d'entrée pour un batch OpenAI.
        
        Args:
            sections_data: Liste des sections à traiter
            corpus_manager: Gestionnaire du corpus
            prompt_builder: Constructeur de prompts
            model: Modèle à utiliser
            corpus_params: Paramètres de filtrage du corpus
            
        Returns:
            ID du fichier uploadé sur OpenAI
        """
        corpus_params = corpus_params or {
            "min_relevance_score": 0.7,
            "max_citations_per_section": 10,
            "include_secondary_matches": True,
            "confidence_threshold": 0.8
        }
        
        # Préparer les requêtes batch
        batch_requests = []
        
        for section in sections_data:
            section_code = section.get('code', '')
            section_title = section.get('title', '')
            
            try:
                # Récupérer le corpus filtré pour cette section
                filtered_corpus = corpus_manager.get_relevant_content(
                    section_title,
                    min_score=corpus_params["min_relevance_score"],
                    max_citations=corpus_params["max_citations_per_section"],
                    include_secondary=corpus_params["include_secondary_matches"],
                    confidence_threshold=corpus_params["confidence_threshold"]
                )
                
                if len(filtered_corpus) == 0:
                    logging.warning(f"Aucune donnée trouvée pour la section '{section_title}'")
                    continue
                
                # Construire le prompt
                prompt = prompt_builder.build_draft_prompt(section_title, filtered_corpus)
                
                # Créer la requête batch
                request = {
                    "custom_id": f"{section_code}_{section_title}".replace(" ", "_"),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 4000,
                        "temperature": 0.7
                    }
                }
                
                batch_requests.append(request)
                
            except Exception as e:
                logging.error(f"Erreur lors de la préparation de la section {section_title}: {e}")
                continue
        
        if not batch_requests:
            raise ValueError("Aucune requête valide préparée pour le batch")
        
        # Créer le fichier d'entrée
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"batch_input_{timestamp}.jsonl"
        input_path = self.batch_files_dir / input_filename
        
        with open(input_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        # Uploader le fichier vers OpenAI
        with open(input_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )
        
        logging.info(f"Fichier batch créé et uploadé: {file_response.id} ({len(batch_requests)} requêtes)")
        return file_response.id
    
    def start_new_batch_process(self, plan_items: List[Dict[str, Any]],
                              corpus_manager: CorpusManager,
                              prompt_builder: PromptBuilder,
                              model: str = "gpt-4o-mini",
                              corpus_params: Dict[str, Any] = None,
                              description: str = "") -> str:
        """
        Lance un nouveau processus de génération par lot.
        
        Args:
            plan_items: Sections à traiter
            corpus_manager: Gestionnaire du corpus
            prompt_builder: Constructeur de prompts
            model: Modèle à utiliser
            corpus_params: Paramètres du corpus
            description: Description du processus
            
        Returns:
            ID du processus créé
        """
        # Créer le processus dans le tracker
        process_id = self.tracker.create_new_process(
            plan_items, 
            process_type="batch", 
            description=description
        )
        
        try:
            # Marquer le processus comme en cours
            self.tracker.update_process_status(process_id, ProcessStatus.EN_COURS)
            
            # Créer le fichier d'entrée
            input_file_id = self.create_batch_input_file(
                plan_items, 
                corpus_manager, 
                prompt_builder, 
                model, 
                corpus_params
            )
            
            # Lancer le batch OpenAI
            batch_response = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "process_id": process_id,
                    "description": description,
                    "model": model
                }
            )
            
            # Enregistrer le batch dans le tracker
            section_codes = [item.get('code', '') for item in plan_items]
            self.tracker.add_batch_to_process(
                process_id, 
                batch_response.id, 
                section_codes, 
                "generation"
            )
            
            # Marquer toutes les sections comme en cours
            for item in plan_items:
                self.tracker.update_section_status(
                    process_id,
                    item.get('code', ''),
                    SectionStatus.EN_COURS,
                    batch_response.id
                )
            
            logging.info(f"Batch lancé: {batch_response.id} pour le processus {process_id}")
            return process_id
            
        except Exception as e:
            # Marquer le processus comme en échec
            self.tracker.update_process_status(process_id, ProcessStatus.EN_ECHEC)
            logging.error(f"Erreur lors du lancement du batch pour le processus {process_id}: {e}")
            raise
    
    def resume_failed_process(self, process_id: str,
                            corpus_manager: CorpusManager,
                            prompt_builder: PromptBuilder,
                            model: str = "gpt-4o-mini",
                            corpus_params: Dict[str, Any] = None) -> Optional[str]:
        """
        Reprend un processus en échec en relançant les sections non terminées.
        
        Args:
            process_id: ID du processus à reprendre
            corpus_manager: Gestionnaire du corpus
            prompt_builder: Constructeur de prompts
            model: Modèle à utiliser
            corpus_params: Paramètres du corpus
            
        Returns:
            ID du nouveau batch ou None si rien à reprendre
        """
        # Récupérer les sections à reprendre
        failed_sections = self.tracker.get_failed_or_pending_sections(process_id)
        
        if not failed_sections:
            logging.info(f"Aucune section à reprendre pour le processus {process_id}")
            return None
        
        try:
            # Convertir les sections en format attendu
            sections_to_resume = [
                {"code": section['code'], "title": section['title']}
                for section in failed_sections
            ]
            
            # Créer un nouveau fichier d'entrée pour les sections en échec
            input_file_id = self.create_batch_input_file(
                sections_to_resume,
                corpus_manager,
                prompt_builder,
                model,
                corpus_params
            )
            
            # Lancer un nouveau batch
            batch_response = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "process_id": process_id,
                    "type": "resume",
                    "model": model
                }
            )
            
            # Enregistrer le nouveau batch
            section_codes = [section['code'] for section in failed_sections]
            self.tracker.add_batch_to_process(
                process_id,
                batch_response.id,
                section_codes,
                "resume"
            )
            
            # Marquer les sections comme en cours
            for section in failed_sections:
                self.tracker.update_section_status(
                    process_id,
                    section['code'],
                    SectionStatus.EN_COURS,
                    batch_response.id
                )
            
            # Remettre le processus en cours
            self.tracker.update_process_status(process_id, ProcessStatus.EN_COURS)
            
            logging.info(f"Reprise lancée: batch {batch_response.id} pour {len(failed_sections)} sections")
            return batch_response.id
            
        except Exception as e:
            logging.error(f"Erreur lors de la reprise du processus {process_id}: {e}")
            raise
    
    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Vérifie le statut d'un batch OpenAI.
        
        Args:
            batch_id: ID du batch à vérifier
            
        Returns:
            Informations sur le statut du batch
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "request_counts": batch.request_counts.__dict__ if batch.request_counts else None,
                "metadata": batch.metadata or {}
            }
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du batch {batch_id}: {e}")
            return {"id": batch_id, "status": "error", "error": str(e)}
    
    def process_batch_results(self, batch_id: str, export_dir: str = "output") -> Dict[str, Any]:
        """
        Traite les résultats d'un batch terminé.
        
        Args:
            batch_id: ID du batch terminé
            export_dir: Dossier de destination pour les exports
            
        Returns:
            Statistiques du traitement
        """
        try:
            # Récupérer le batch
            batch = self.client.batches.retrieve(batch_id)
            
            if batch.status != "completed":
                raise ValueError(f"Le batch {batch_id} n'est pas terminé (statut: {batch.status})")
            
            if not batch.output_file_id:
                raise ValueError(f"Aucun fichier de sortie disponible pour le batch {batch_id}")
            
            # Télécharger les résultats
            output_file = self.client.files.content(batch.output_file_id)
            results_content = output_file.read().decode('utf-8')
            
            # Parser les résultats
            results = []
            for line in results_content.strip().split('\n'):
                if line:
                    results.append(json.loads(line))
            
            # Traiter chaque résultat
            process_id = batch.metadata.get('process_id') if batch.metadata else None
            success_count = 0
            error_count = 0
            
            os.makedirs(export_dir, exist_ok=True)
            
            for result in results:
                custom_id = result.get('custom_id', '')
                
                # Extraire le code de section du custom_id
                section_code = custom_id.split('_')[0] if '_' in custom_id else custom_id
                
                if result.get('response') and result['response'].get('body'):
                    # Succès
                    response_body = result['response']['body']
                    if response_body.get('choices') and len(response_body['choices']) > 0:
                        generated_text = response_body['choices'][0]['message']['content']
                        
                        # Sauvegarder le résultat
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_filename = f"{section_code}_{timestamp}_batch_result.md"
                        result_path = os.path.join(export_dir, result_filename)
                        
                        with open(result_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {custom_id}\n\n")
                            f.write(f"Généré le : {datetime.now().isoformat()}\n")
                            f.write(f"Batch ID : {batch_id}\n\n")
                            f.write("---\n\n")
                            f.write(generated_text)
                        
                        # Mettre à jour le tracker si disponible
                        if process_id:
                            self.tracker.update_section_status(
                                process_id,
                                section_code,
                                SectionStatus.SUCCES,
                                batch_id,
                                result_path=result_path
                            )
                        
                        success_count += 1
                    else:
                        error_count += 1
                        if process_id:
                            self.tracker.update_section_status(
                                process_id,
                                section_code,
                                SectionStatus.ECHEC,
                                batch_id,
                                error_message="Réponse vide du modèle"
                            )
                else:
                    # Erreur
                    error_message = result.get('error', {}).get('message', 'Erreur inconnue')
                    error_count += 1
                    
                    if process_id:
                        self.tracker.update_section_status(
                            process_id,
                            section_code,
                            SectionStatus.ECHEC,
                            batch_id,
                            error_message=error_message
                        )
            
            logging.info(f"Batch {batch_id} traité: {success_count} succès, {error_count} erreurs")
            
            return {
                "batch_id": batch_id,
                "total_requests": len(results),
                "success_count": success_count,
                "error_count": error_count,
                "export_dir": export_dir
            }
            
        except Exception as e:
            logging.error(f"Erreur lors du traitement des résultats du batch {batch_id}: {e}")
            raise
    
    def monitor_processes(self, callback: Callable[[str, Dict[str, Any]], None] = None) -> List[Dict[str, Any]]:
        """
        Surveille tous les processus en cours et met à jour leur statut.
        
        Args:
            callback: Fonction de callback appelée pour chaque mise à jour
            
        Returns:
            Liste des processus mis à jour
        """
        updated_processes = []
        
        # Récupérer tous les processus en cours
        all_processes = self.tracker.get_all_processes()
        in_progress_processes = [
            p for p in all_processes 
            if p['status'] == ProcessStatus.EN_COURS.value
        ]
        
        for process in in_progress_processes:
            process_id = process['process_id']
            updated = False
            
            # Vérifier chaque batch du processus
            for batch_info in process.get('batch_history', []):
                batch_id = batch_info['batch_id']
                
                try:
                    # Vérifier le statut du batch
                    batch_status = self.check_batch_status(batch_id)
                    
                    if batch_status['status'] == 'completed':
                        # Traiter les résultats si pas encore fait
                        if batch_info.get('status') != 'processed':
                            self.process_batch_results(batch_id)
                            batch_info['status'] = 'processed'
                            updated = True
                    
                    elif batch_status['status'] in ['failed', 'expired', 'cancelled']:
                        # Marquer les sections de ce batch comme en échec
                        for section_code in batch_info.get('section_codes', []):
                            self.tracker.update_section_status(
                                process_id,
                                section_code,
                                SectionStatus.ECHEC,
                                batch_id,
                                error_message=f"Batch {batch_status['status']}"
                            )
                        batch_info['status'] = 'failed'
                        updated = True
                
                except Exception as e:
                    logging.error(f"Erreur lors de la surveillance du batch {batch_id}: {e}")
            
            if updated:
                updated_processes.append(process)
                if callback:
                    callback(process_id, process)
        
        return updated_processes
    
    def get_process_status_summary(self, process_id: str) -> Dict[str, Any]:
        """
        Retourne un résumé détaillé du statut d'un processus.
        
        Args:
            process_id: ID du processus
            
        Returns:
            Résumé du statut
        """
        process = self.tracker.get_process(process_id)
        if not process:
            return {"error": "Processus non trouvé"}
        
        summary = self.tracker.get_process_summary(process_id)
        
        # Ajouter des informations sur les batchs
        batch_info = []
        for batch in process.get('batch_history', []):
            batch_status = self.check_batch_status(batch['batch_id'])
            batch_info.append({
                "batch_id": batch['batch_id'],
                "type": batch.get('batch_type', 'generation'),
                "status": batch_status['status'],
                "section_count": len(batch.get('section_codes', [])),
                "created_at": batch.get('created_at')
            })
        
        summary['batch_history'] = batch_info
        return summary
