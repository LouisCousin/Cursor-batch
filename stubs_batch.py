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

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    logging.warning("Anthropic non disponible. Fonctionnalités de batch désactivées pour ce fournisseur.")

# Import des modules locaux
import sys
sys.path.append('src')

from core.process_tracker import ProcessTracker, ProcessStatus, SectionStatus
from core.prompt_builder import PromptBuilder
from core.corpus_manager import CorpusManager
from config_manager import get_model_config
from converter import convert_md_to_docx
from core.anthropic_batch_processor import AnthropicBatchProcessor


class BatchProcessor:
    """
    Gestionnaire des traitements par lot via les API Batch OpenAI ou Anthropic.
    """

    def __init__(self, api_key: str, provider: str, process_tracker: ProcessTracker = None):
        """
        Initialise le processeur de batch.

        Args:
            api_key: Clé API pour le fournisseur sélectionné.
            provider: Nom du fournisseur ("OpenAI" ou "Anthropic").
            process_tracker: Tracker de processus (optionnel).
        """
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("La clé API fournie au BatchProcessor est invalide (vide ou nulle).")

        if provider == "OpenAI":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI non disponible. Impossible d'utiliser le traitement par lot.")
            self.client = OpenAI(api_key=api_key)
        elif provider == "Anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic non disponible. Impossible d'utiliser le traitement par lot.")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Fournisseur non supporté pour le traitement par lot : {provider}")

        self.provider = provider
        self.api_key = api_key
        self.tracker = process_tracker or ProcessTracker()
        self.batch_files_dir = Path("data/batch_files")
        self.batch_files_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_specific_params(self, model: str) -> Dict[str, Any]:
        """
        Retourne les paramètres appropriés selon le modèle utilisé.
        Utilise la configuration depuis config_manager pour les limites de tokens.
        
        Args:
            model: Nom du modèle (ex: gpt-5, gpt-4.1, gpt-4o-mini, etc.)
            
        Returns:
            Dictionnaire des paramètres appropriés
        """
        # Récupérer la configuration du modèle
        model_config = get_model_config(model)
        params = {}
        
        # Utiliser max_output_tokens du fichier de configuration
        if 'max_output' in model_config:
            # Détecter le type de modèle selon la documentation GPT-5
            if model.lower() in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']:
                # Modèles de raisonnement GPT-5 : utilisent max_completion_tokens, PAS temperature
                params.update({
                    'max_completion_tokens': model_config['max_output'],
                    # PAS de temperature/top_p pour modèles de raisonnement
                    'reasoning_effort': 'medium',  # Paramètre spécifique GPT-5
                    'verbosity': 'medium'  # Paramètre spécifique GPT-5
                })
            elif 'gpt-5-chat' in model.lower():
                # gpt-5-chat-latest : modèle non-raisonnement, utilise max_tokens
                params.update({
                    'max_tokens': model_config['max_output'],
                    'temperature': 0.7,  # Supporté par gpt-5-chat
                    'top_p': 1.0
                    # PAS de reasoning_effort/verbosity pour gpt-5-chat
                })
            elif model.startswith('gpt-4.1'):
                # GPT-4.1 : utilise max_tokens avec capacité de contexte étendue
                params.update({
                    'max_tokens': model_config['max_output'],
                    'temperature': 0.7,  # Flexible comme GPT-4
                    'top_p': 1.0
                })
            else:
                # Modèles GPT-4 et antérieurs : utiliser max_tokens
                params.update({
                    'max_tokens': model_config['max_output'],
                    'temperature': 0.7,  # Flexible pour GPT-4 et antérieurs
                    'top_p': 1.0
                })
            
        return params
        
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
            "min_relevance_score": 1,
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
                
                # Créer la requête batch avec paramètres adaptés au modèle
                model_params = self.get_model_specific_params(model)
                
                # Filtrage spécifique pour les modèles GPT-5 de raisonnement
                model_name = model.lower()
                is_gpt5_reasoning_model = "gpt-5" in model_name and "chat" not in model_name
                
                # Créer une copie pour éviter de modifier le dictionnaire original
                request_body_params = model_params.copy()
                
                if is_gpt5_reasoning_model:
                    # Pour les modèles GPT-5 de raisonnement, retirer les paramètres non supportés
                    request_body_params.pop("temperature", None)
                    request_body_params.pop("top_p", None)
                    # Assurer que les paramètres spécifiques à GPT-5 sont présents
                    request_body_params.setdefault("verbosity", "medium")
                    request_body_params.setdefault("reasoning_effort", "medium")
                
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
                        **request_body_params
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
            description=description,
            provider=self.provider.lower(),
            model_name=model
        )
        
        try:
            # Marquer le processus comme en cours
            self.tracker.update_process_status(process_id, ProcessStatus.EN_COURS)

            if self.provider == "OpenAI":
                # Logique OpenAI existante
                input_file_id = self.create_batch_input_file(
                    plan_items,
                    corpus_manager,
                    prompt_builder,
                    model,
                    corpus_params
                )
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
                batch_id = batch_response.id

            elif self.provider == "Anthropic":
                # Nouvelle logique pour Anthropic
                anthropic_processor = AnthropicBatchProcessor(api_key=self.api_key)  # Pass API key explicitly to avoid auth errors

                prompts_data = []
                for item in plan_items:
                    filtered_corpus = corpus_manager.get_relevant_content(
                        item['title'], **(corpus_params or {})
                    )
                    prompt_content = prompt_builder.build_draft_prompt(
                        item['title'], filtered_corpus
                    )
                    prompts_data.append({
                        "content": prompt_content,
                        "section_code": item['code']
                    })

                anthropic_requests = anthropic_processor.prepare_batch_requests(
                    prompts_data, model
                )
                batch_id = anthropic_processor.launch_batch(anthropic_requests)

            else:
                raise ValueError(
                    f"Logique de lancement non implémentée pour le fournisseur : {self.provider}"
                )

            # Enregistrer le batch dans le tracker
            section_codes = [item.get('code', '') for item in plan_items]
            self.tracker.add_batch_to_process(
                process_id,
                batch_id,
                section_codes,
                "generation",
                provider=self.provider.lower()
            )

            # Marquer toutes les sections comme en cours
            for item in plan_items:
                self.tracker.update_section_status(
                    process_id,
                    item.get('code', ''),
                    SectionStatus.EN_COURS,
                    batch_id
                )

            logging.info(
                f"Batch {self.provider} lancé: {batch_id} pour le processus {process_id}"
            )
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
                "resume",
                provider=self.provider.lower()
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
                # Vérifier s'il y a un fichier d'erreur
                if batch.error_file_id:
                    try:
                        error_content_bytes = self.client.files.content(batch.error_file_id)
                        error_content = error_content_bytes.decode('utf-8')
                        
                        # Extraire le message d'erreur précis de la première ligne du fichier JSONL
                        error_message = f"Le batch a échoué avec des erreurs."
                        try:
                            first_error_line = error_content.split('\n')[0]
                            if first_error_line:
                                parsed_error = json.loads(first_error_line)
                                # Naviguer dans la structure pour trouver le message d'erreur précis
                                api_error_msg = parsed_error.get('response', {}).get('body', {}).get('error', {}).get('message')
                                if api_error_msg:
                                    error_message = f"Erreur API OpenAI : {api_error_msg}"
                                else:
                                    # Essayer une autre structure possible
                                    error_obj = parsed_error.get('error', {})
                                    if error_obj.get('message'):
                                        error_message = f"Erreur API : {error_obj['message']}"
                                    else:
                                        error_message = "Fichier d'erreur du batch présent mais message illisible."
                            else:
                                error_message = "Fichier d'erreur du batch est vide."
                        except Exception as parse_error:
                            logging.warning(f"Impossible de parser le fichier d'erreur : {parse_error}")
                            error_message = f"Impossible de parser le fichier d'erreur du batch : {parse_error}"
                        
                        logging.error(f"Batch {batch_id} terminé avec erreurs: {error_message}")
                        
                        # Récupérer le nombre réel de requêtes depuis les métadonnées du batch
                        total_requests = 0
                        if hasattr(batch, 'request_counts') and batch.request_counts:
                            total_requests = batch.request_counts.total or 0
                        
                        # Marquer toutes les sections comme en échec si nous avons un process_id
                        process_id = batch.metadata.get('process_id') if batch.metadata else None
                        if process_id:
                            # Récupérer toutes les sections de ce batch depuis le tracker
                            process = self.tracker.get_process(process_id)
                            if process:
                                for batch_info in process.get('batch_history', []):
                                    if batch_info['batch_id'] == batch_id:
                                        for section_code in batch_info.get('section_codes', []):
                                            self.tracker.update_section_status(
                                                process_id,
                                                section_code,
                                                SectionStatus.ECHEC,
                                                batch_id,
                                                error_message=error_message
                                            )
                        
                        return {
                            "batch_id": batch_id,
                            "total_requests": total_requests,
                            "success_count": 0,
                            "error_count": total_requests,  # Toutes les requêtes ont échoué
                            "error_message": "Batch terminé avec erreurs - aucun résultat généré",
                            "export_dir": export_dir
                        }
                    except Exception as e:
                        logging.error(f"Impossible de lire le fichier d'erreur du batch {batch_id}: {e}")
                
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
                        
                        # Construire le contenu complet du fichier
                        full_content = f"# {custom_id}\n\n"
                        full_content += f"Généré le : {datetime.now().isoformat()}\n"
                        full_content += f"Batch ID : {batch_id}\n\n"
                        full_content += "---\n\n"
                        full_content += generated_text
                        
                        # Sauvegarder le fichier MD
                        with open(result_path, 'w', encoding='utf-8') as f:
                            f.write(full_content)
                        
                        # Convertir en chemin absolu
                        absolute_result_path = os.path.abspath(result_path)
                        
                        # Créer également un fichier Docx
                        try:
                            docx_path = result_path.replace(".md", ".docx")
                            convert_md_to_docx(full_content, docx_path)
                            logging.info(f"Fichier Docx créé : {docx_path}")
                        except Exception as e:
                            logging.warning(f"Erreur lors de la conversion en DOCX pour {result_path}: {e}")
                        
                        # Mettre à jour le tracker si disponible
                        if process_id:
                            self.tracker.update_section_status(
                                process_id,
                                section_code,
                                SectionStatus.SUCCES,
                                batch_id,
                                result_path=absolute_result_path
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
                            # Vérifier s'il y a un fichier de sortie avant de traiter
                            batch_details = self.client.batches.retrieve(batch_id)
                            
                            if batch_details.output_file_id:
                                try:
                                    self.process_batch_results(batch_id, export_dir="data/output")
                                    batch_info['status'] = 'processed'
                                    updated = True
                                except Exception as e:
                                    logging.error(f"Erreur lors du traitement du batch {batch_id}: {e}")
                                    batch_info['status'] = 'failed_processing'
                                    updated = True
                            else:
                                # Batch terminé mais sans fichier de sortie - ignorer
                                logging.warning(f"Batch {batch_id} terminé sans fichier de sortie - ignoré")
                                batch_info['status'] = 'failed_no_output'
                                updated = True
                    
                    elif batch_status['status'] in ['failed', 'expired', 'cancelled']:
                        # Récupérer les détails d'erreur si disponible
                        detailed_error_message = f"Batch {batch_status['status']}"
                        
                        try:
                            # Essayer de récupérer le batch complet pour obtenir error_file_id
                            batch_details = self.client.batches.retrieve(batch_id)
                            if batch_details.error_file_id:
                                error_content_bytes = self.client.files.content(batch_details.error_file_id)
                                error_content = error_content_bytes.decode('utf-8')
                                
                                # Extraire le message d'erreur précis
                                try:
                                    first_error_line = error_content.split('\n')[0]
                                    if first_error_line:
                                        parsed_error = json.loads(first_error_line)
                                        api_error_msg = parsed_error.get('response', {}).get('body', {}).get('error', {}).get('message')
                                        if api_error_msg:
                                            detailed_error_message = f"Erreur API OpenAI : {api_error_msg}"
                                        else:
                                            error_obj = parsed_error.get('error', {})
                                            if error_obj.get('message'):
                                                detailed_error_message = f"Erreur API : {error_obj['message']}"
                                except Exception:
                                    # Si on ne peut pas parser, garder le message générique
                                    pass
                        except Exception as e:
                            logging.warning(f"Impossible de récupérer les détails d'erreur pour le batch {batch_id}: {e}")
                        
                        # Marquer les sections de ce batch comme en échec avec le message détaillé
                        for section_code in batch_info.get('section_codes', []):
                            self.tracker.update_section_status(
                                process_id,
                                section_code,
                                SectionStatus.ECHEC,
                                batch_id,
                                error_message=detailed_error_message
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
    
    def auto_process_completed_batches(self, export_dir: str = "data/output") -> Dict[str, Any]:
        """
        Traite automatiquement tous les batches terminés qui n'ont pas encore été traités.
        
        Args:
            export_dir: Dossier de destination pour les exports (optionnel)
        
        Returns:
            Statistiques du traitement automatique
        """
        stats = {
            "processed_batches": 0,
            "total_sections": 0,
            "successful_sections": 0,
            "failed_sections": 0,
            "errors": []
        }
        
        # Récupérer tous les processus
        all_processes = self.tracker.get_all_processes()
        
        for process in all_processes:
            process_id = process['process_id']
            
            # Vérifier chaque batch du processus
            for batch_info in process.get('batch_history', []):
                batch_id = batch_info['batch_id']
                
                # Vérifier si le batch est terminé mais pas encore traité
                if batch_info.get('status') != 'processed':
                    try:
                        batch_status = self.check_batch_status(batch_id)
                        
                        if batch_status['status'] == 'completed':
                            # Vérifier s'il y a des résultats utilisables avant de traiter
                            batch_details = self.client.batches.retrieve(batch_id)
                            
                            # Ne traiter que les batches avec un fichier de sortie
                            if batch_details.output_file_id:
                                # Traiter les résultats
                                result = self.process_batch_results(batch_id, export_dir=export_dir)
                                
                                # Mettre à jour les statistiques
                                stats["processed_batches"] += 1
                                stats["total_sections"] += result.get("total_requests", 0)
                                stats["successful_sections"] += result.get("success_count", 0)
                                
                                # error_count est maintenant toujours un entier
                                stats["failed_sections"] += result.get("error_count", 0)
                                
                                # Marquer comme traité
                                batch_info['status'] = 'processed'
                                
                                logging.info(f"Batch {batch_id} traité automatiquement")
                            else:
                                # Batch terminé mais sans résultats - marquer comme échec
                                batch_info['status'] = 'failed_no_output'
                                logging.warning(f"Batch {batch_id} terminé sans fichier de sortie - ignoré")
                            
                    except Exception as e:
                        error_msg = f"Erreur lors du traitement automatique du batch {batch_id}: {e}"
                        logging.error(error_msg)
                        stats["errors"].append(error_msg)
        
        return stats
    
    def get_batch_completion_estimate(self, batch_id: str) -> Dict[str, Any]:
        """
        Fournit une estimation du temps de completion et des progrès d'un batch.
        
        Args:
            batch_id: ID du batch à analyser
            
        Returns:
            Estimation de completion avec pourcentage et temps restant
        """
        try:
            batch_status = self.check_batch_status(batch_id)
            
            if batch_status.get('status') == 'error':
                return {"error": "Impossible de récupérer le statut du batch"}
            
            request_counts = batch_status.get('request_counts', {})
            if not request_counts:
                return {"error": "Informations de progression non disponibles"}
            
            total = request_counts.get('total', 0)
            completed = request_counts.get('completed', 0)
            failed = request_counts.get('failed', 0)
            
            if total == 0:
                return {"error": "Aucune requête dans le batch"}
            
            # Calcul du pourcentage
            processed = completed + failed
            completion_percentage = (processed / total) * 100
            
            # Estimation du temps restant
            created_at = batch_status.get('created_at')
            if created_at and processed > 0:
                from datetime import datetime
                start_time = datetime.fromtimestamp(created_at)
                elapsed_time = datetime.now() - start_time
                
                if completion_percentage > 0:
                    estimated_total_time = elapsed_time * (100 / completion_percentage)
                    remaining_time = estimated_total_time - elapsed_time
                    remaining_minutes = int(remaining_time.total_seconds() / 60)
                else:
                    remaining_minutes = "Impossible à estimer"
            else:
                remaining_minutes = "Impossible à estimer"
            
            return {
                "batch_id": batch_id,
                "status": batch_status.get('status'),
                "progress": {
                    "total_requests": total,
                    "completed": completed,
                    "failed": failed,
                    "pending": total - processed,
                    "completion_percentage": round(completion_percentage, 1)
                },
                "time_estimate": {
                    "remaining_minutes": remaining_minutes,
                    "status": "En cours" if batch_status.get('status') == 'in_progress' else batch_status.get('status')
                }
            }
            
        except Exception as e:
            return {"error": f"Erreur lors de l'estimation: {e}"}
    
    def diagnose_batch_issues(self, batch_id: str) -> Dict[str, Any]:
        """
        Effectue un diagnostic complet d'un batch pour identifier les problèmes.
        
        Args:
            batch_id: ID du batch à diagnostiquer
            
        Returns:
            Rapport de diagnostic détaillé
        """
        try:
            # Récupérer les informations du batch
            batch = self.client.batches.retrieve(batch_id)
            
            diagnosis = {
                "batch_id": batch_id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "request_counts": batch.request_counts.__dict__ if batch.request_counts else None,
                "issues": [],
                "recommendations": []
            }
            
            # Vérifier les problèmes courants
            if batch.status == "completed" and not batch.output_file_id:
                diagnosis["issues"].append("Batch terminé mais sans fichier de sortie")
                diagnosis["recommendations"].append("Vérifier le fichier d'erreur pour les détails")
                
                # Vérifier le fichier d'erreur
                if batch.error_file_id:
                    try:
                        error_file = self.client.files.content(batch.error_file_id)
                        error_content = error_file.read().decode('utf-8')
                        
                        # Analyser les erreurs communes selon documentation GPT-5
                        if "max_tokens" in error_content and "not supported" in error_content and "max_completion_tokens" in error_content:
                            diagnosis["issues"].append("Modèle de raisonnement GPT-5 détecté : max_tokens → max_completion_tokens")
                            diagnosis["recommendations"].append("Utiliser max_completion_tokens pour gpt-5/gpt-5-mini/gpt-5-nano (corrigé automatiquement)")
                        
                        if "max_output_tokens" in error_content and "Unknown parameter" in error_content:
                            diagnosis["issues"].append("Paramètre max_output_tokens non reconnu par ce modèle")
                            diagnosis["recommendations"].append("Paramètres mis à jour selon documentation officielle GPT-5")
                        
                        if "temperature" in error_content and ("does not support" in error_content or "not supported" in error_content):
                            diagnosis["issues"].append("Modèle de raisonnement GPT-5 : temperature non supporté")
                            diagnosis["recommendations"].append("Utiliser gpt-5-chat-latest si temperature requis, sinon reasoning_effort/verbosity")
                        
                        # Ajouter un échantillon des erreurs
                        error_lines = error_content.strip().split('\n')[:3]  # 3 premières lignes
                        diagnosis["error_sample"] = error_lines
                        
                    except Exception as e:
                        diagnosis["issues"].append(f"Impossible de lire le fichier d'erreur: {e}")
            
            elif batch.status == "failed":
                diagnosis["issues"].append("Batch en échec")
                diagnosis["recommendations"].append("Relancer le batch avec des paramètres corrigés")
            
            elif batch.status == "expired":
                diagnosis["issues"].append("Batch expiré (>24h)")
                diagnosis["recommendations"].append("Relancer un nouveau batch")
            
            # Vérifier le taux d'échec
            if batch.request_counts:
                total = batch.request_counts.total or 0
                failed = batch.request_counts.failed or 0
                
                if total > 0:
                    failure_rate = (failed / total) * 100
                    
                    if failure_rate > 50:
                        diagnosis["issues"].append(f"Taux d'échec élevé: {failure_rate:.1f}%")
                        diagnosis["recommendations"].append("Vérifier les paramètres du modèle et les prompts")
                    
                    diagnosis["failure_rate"] = round(failure_rate, 1)
            
            return diagnosis
            
        except Exception as e:
            return {
                "batch_id": batch_id,
                "error": f"Erreur lors du diagnostic: {e}",
                "issues": ["Impossible d'accéder aux informations du batch"],
                "recommendations": ["Vérifier que le batch_id est correct et que vous avez les permissions"]
            }
