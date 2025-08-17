#!/usr/bin/env python3
"""
Module de gestion des batchs Anthropic.
Gère le cycle de vie complet des batchs Anthropic : création, suivi, récupération des résultats.
"""

import uuid
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicBatchProcessor:
    """
    Gestionnaire des batchs Anthropic avec suivi de statut et récupération des résultats.
    """
    
    def __init__(self, api_key: str):
        """
        Initialise le processeur de batch Anthropic.
        
        Args:
            api_key: Clé API Anthropic
        """
        if not anthropic:
            raise ImportError("Le module anthropic n'est pas installé. Installez-le avec: pip install anthropic")
        
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.base_url = "https://api.anthropic.com/v1"
        
        logging.info("AnthropicBatchProcessor initialisé")
    
    def prepare_batch_requests(self, prompts: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        Prépare les requêtes pour un batch Anthropic.
        
        Args:
            prompts: Liste des prompts avec leurs métadonnées
            model_name: Nom du modèle Anthropic à utiliser
            
        Returns:
            Liste des requêtes formatées pour l'API Anthropic
        """
        requests = []
        
        for prompt_data in prompts:
            custom_id = f"request-{uuid.uuid4()}"
            
            # Extraire le contenu du prompt
            if isinstance(prompt_data, dict):
                content = prompt_data.get('content', '')
                section_code = prompt_data.get('section_code', '')
                max_tokens = prompt_data.get('max_tokens', 4096)
                temperature = prompt_data.get('temperature', 0.7)
                top_p = prompt_data.get('top_p', 0.9)
            else:
                # Si c'est juste une string
                content = str(prompt_data)
                section_code = f"section-{len(requests)}"
                max_tokens = 4096
                temperature = 0.7
                top_p = 0.9
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
            # Ajouter les métadonnées pour la réconciliation
            request["metadata"] = {
                "section_code": section_code,
                "custom_id": custom_id
            }
            
            requests.append(request)
            
            logging.debug(f"Requête préparée pour section {section_code}: {custom_id}")
        
        logging.info(f"Préparation de {len(requests)} requêtes pour le batch Anthropic")
        return requests
    
    def launch_batch(self, requests: List[Dict[str, Any]]) -> str:
        """
        Lance un batch Anthropic.
        
        Args:
            requests: Liste des requêtes préparées
            
        Returns:
            ID du batch créé
            
        Raises:
            Exception: En cas d'erreur lors de la création du batch
        """
        try:
            # Préparer la payload sans les métadonnées pour l'API
            api_requests = []
            for req in requests:
                api_request = {
                    "custom_id": req["custom_id"],
                    "params": req["params"]
                }
                api_requests.append(api_request)
            
            # Créer le batch via l'API Anthropic
            message_batch = self.client.messages.batches.create(
                requests=api_requests
            )
            
            batch_id = message_batch.id
            logging.info(f"Batch Anthropic créé avec succès: {batch_id}")
            
            return batch_id
            
        except Exception as e:
            logging.error(f"Erreur lors de la création du batch Anthropic: {e}")
            raise Exception(f"Impossible de créer le batch Anthropic: {e}")
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Récupère le statut d'un batch Anthropic.
        
        Args:
            batch_id: ID du batch
            
        Returns:
            Dictionnaire avec les informations de statut
        """
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
            
            # Convertir request_counts en dictionnaire pour une utilisation cohérente
            request_counts_obj = getattr(batch, 'request_counts', None)
            request_counts_dict = {}
            if request_counts_obj:
                request_counts_dict = {
                    "processing": getattr(request_counts_obj, 'processing', 0),
                    "succeeded": getattr(request_counts_obj, 'succeeded', 0),
                    "errored": getattr(request_counts_obj, 'errored', 0),
                    "canceled": getattr(request_counts_obj, 'canceled', 0),
                    "expired": getattr(request_counts_obj, 'expired', 0)
                }
            
            status_info = {
                "id": batch.id,
                "processing_status": batch.processing_status,
                "request_counts": request_counts_dict,
                "created_at": getattr(batch, 'created_at', None),
                "expires_at": getattr(batch, 'expires_at', None),
                "results_url": getattr(batch, 'results_url', None)
            }
            
            logging.debug(f"Statut du batch {batch_id}: {status_info['processing_status']}")
            return status_info
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du statut du batch {batch_id}: {e}")
            return {
                "id": batch_id,
                "processing_status": "error",
                "error": str(e)
            }
    
    def is_batch_completed(self, batch_id: str) -> bool:
        """
        Vérifie si un batch est terminé.
        
        Args:
            batch_id: ID du batch
            
        Returns:
            True si le batch est terminé (succès ou échec)
        """
        status_info = self.get_batch_status(batch_id)
        processing_status = status_info.get("processing_status")
        
        return processing_status in ["ended", "failed", "expired", "cancelled"]
    
    def download_results(self, results_url: str) -> List[Dict[str, Any]]:
        """
        Télécharge et parse les résultats d'un batch depuis l'URL fournie.
        
        Args:
            results_url: URL des résultats fournie par l'API Anthropic
            
        Returns:
            Liste des résultats parsés
        """
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.get(results_url, headers=headers)
            response.raise_for_status()
            
            # Parser le fichier JSONL
            results = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Impossible de parser la ligne: {line[:100]}... Erreur: {e}")
            
            logging.info(f"Téléchargement réussi: {len(results)} résultats")
            return results
            
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des résultats: {e}")
            raise Exception(f"Impossible de télécharger les résultats: {e}")
    
    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Récupère les résultats d'un batch terminé.
        
        Args:
            batch_id: ID du batch
            
        Returns:
            Liste des résultats avec le contenu généré
            
        Raises:
            Exception: Si le batch n'est pas terminé ou en cas d'erreur
        """
        status_info = self.get_batch_status(batch_id)
        
        if status_info.get("processing_status") != "ended":
            raise Exception(f"Le batch {batch_id} n'est pas terminé. Statut: {status_info.get('processing_status')}")
        
        results_url = status_info.get("results_url")
        if not results_url:
            raise Exception(f"URL des résultats non disponible pour le batch {batch_id}")
        
        raw_results = self.download_results(results_url)
        
        # Traiter les résultats pour extraire le contenu généré
        processed_results = []
        for result in raw_results:
            try:
                custom_id = result.get("custom_id")
                result_data = result.get("result", {})
                
                # Extraire le contenu de la réponse Anthropic
                content = ""
                if "message" in result_data:
                    message = result_data["message"]
                    if "content" in message:
                        for content_block in message["content"]:
                            if content_block.get("type") == "text":
                                content += content_block.get("text", "")
                
                processed_result = {
                    "custom_id": custom_id,
                    "content": content,
                    "raw_result": result_data
                }
                
                # Ajouter les informations d'erreur si présentes
                if "error" in result:
                    processed_result["error"] = result["error"]
                
                processed_results.append(processed_result)
                
            except Exception as e:
                logging.warning(f"Erreur lors du traitement du résultat {result.get('custom_id', 'unknown')}: {e}")
                processed_results.append({
                    "custom_id": result.get("custom_id", "unknown"),
                    "content": "",
                    "error": f"Erreur de traitement: {e}",
                    "raw_result": result
                })
        
        logging.info(f"Traitement terminé: {len(processed_results)} résultats pour le batch {batch_id}")
        return processed_results
    
    def wait_for_completion(self, batch_id: str, max_wait_time: int = 3600, check_interval: int = 30) -> bool:
        """
        Attend qu'un batch soit terminé.
        
        Args:
            batch_id: ID du batch
            max_wait_time: Temps d'attente maximum en secondes (défaut: 1 heure)
            check_interval: Intervalle entre les vérifications en secondes (défaut: 30s)
            
        Returns:
            True si le batch est terminé avec succès, False sinon
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if self.is_batch_completed(batch_id):
                status_info = self.get_batch_status(batch_id)
                success = status_info.get("processing_status") == "ended"
                
                if success:
                    logging.info(f"Batch {batch_id} terminé avec succès")
                else:
                    logging.warning(f"Batch {batch_id} terminé avec erreur: {status_info.get('processing_status')}")
                
                return success
            
            logging.debug(f"Batch {batch_id} en cours, attente de {check_interval}s...")
            time.sleep(check_interval)
        
        logging.warning(f"Timeout atteint pour le batch {batch_id} après {max_wait_time}s")
        return False


def launch_anthropic_batch(prompts: List[Dict[str, Any]], model_name: str, api_key: str) -> str:
    """
    Fonction utilitaire pour lancer un batch Anthropic.
    
    Args:
        prompts: Liste des prompts à traiter
        model_name: Nom du modèle Anthropic
        api_key: Clé API Anthropic
        
    Returns:
        ID du batch créé
    """
    processor = AnthropicBatchProcessor(api_key)
    requests = processor.prepare_batch_requests(prompts, model_name)
    return processor.launch_batch(requests)


def get_anthropic_batch_status(batch_id: str, api_key: str) -> Dict[str, Any]:
    """
    Fonction utilitaire pour récupérer le statut d'un batch Anthropic.
    
    Args:
        batch_id: ID du batch
        api_key: Clé API Anthropic
        
    Returns:
        Informations de statut du batch
    """
    processor = AnthropicBatchProcessor(api_key)
    return processor.get_batch_status(batch_id)


def get_anthropic_batch_results(batch_id: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fonction utilitaire pour récupérer les résultats d'un batch Anthropic.
    
    Args:
        batch_id: ID du batch
        api_key: Clé API Anthropic
        
    Returns:
        Liste des résultats du batch
    """
    processor = AnthropicBatchProcessor(api_key)
    return processor.get_batch_results(batch_id)
