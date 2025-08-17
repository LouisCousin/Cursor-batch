#!/usr/bin/env python3
"""
Test rapide pour vérifier le statut du batch Anthropic avec notre correction.
"""

import sys
import os

# Ajouter le chemin
sys.path.insert(0, 'src')

from core.anthropic_batch_processor import get_anthropic_batch_status

def test_batch_status():
    """Test du statut du batch avec votre ID réel."""
    
    # Votre batch ID
    batch_id = "msgbatch_01FQR2XGLmeCJ168kQkC3b2F"
    
    print(f"🔍 Test du batch : {batch_id}")
    
    # Récupérer la clé API (à vous de la mettre)
    api_key = input("Entrez votre clé API Anthropic: ").strip()
    
    if not api_key:
        print("❌ Clé API requise")
        return
    
    try:
        # Tester notre fonction corrigée
        status = get_anthropic_batch_status(batch_id, api_key)
        
        print("\n✅ Statut récupéré avec succès !")
        print(f"📊 Statut: {status.get('processing_status')}")
        print(f"🔢 Request counts: {status.get('request_counts')}")
        
        # Tester l'utilisation de .get() (qui causait l'erreur avant)
        counts = status.get('request_counts', {})
        total = counts.get('processing', 0) + counts.get('succeeded', 0) + counts.get('errored', 0)
        completed = counts.get('succeeded', 0) + counts.get('errored', 0)
        
        print(f"✅ Total: {total}, Complété: {completed}")
        print("🎉 Plus d'erreur 'MessageBatchRequestCounts' object has no attribute 'get' !")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    test_batch_status()
