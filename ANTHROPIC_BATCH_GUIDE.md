# Guide d'utilisation du Mode Batch Anthropic

## Vue d'ensemble

Le système prend désormais en charge l'API Batch d'Anthropic en plus de l'API Batch d'OpenAI, permettant de réduire les coûts de 50% pour les générations utilisant les modèles Anthropic tout en maintenant une expérience utilisateur transparente.

## Configuration

### 1. Ajout de la clé API Anthropic

Modifiez le fichier `config/user.yaml` pour ajouter votre clé API Anthropic :

```yaml
# Clés d'API
api_keys:
  openai_api_key: "sk-..."      # Votre clé API OpenAI
  anthropic_api_key: "sk-ant-..."  # Votre clé API Anthropic
```

Alternativement, vous pouvez définir la variable d'environnement :
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Modèles supportés

Les modèles Anthropic suivants sont pris en charge pour le mode batch :

- `claude-sonnet-4-20250514` (recommandé)
- `claude-3.5-sonnet-20240620`

## Utilisation

### Lancement d'un batch Anthropic

1. **Sélection du modèle** : Dans l'interface utilisateur, sélectionnez un modèle Anthropic comme modèle de brouillon
2. **Mode de traitement** : Choisissez "Batch (traitement différé)"
3. **Génération** : Cliquez sur "Lancer la Génération"

Le système détecte automatiquement le fournisseur du modèle et utilise l'API appropriée (OpenAI ou Anthropic).

### Suivi des batchs

Dans l'onglet "Historique des Générations" :

- **Statut du processus** : Affiche le fournisseur et le modèle utilisé (ex: "Anthropic: claude-sonnet-4-20250514")
- **Historique des lots** : Indique le fournisseur de chaque batch
- **Actions disponibles** :
  - 🔍 **Vérifier** : Consulte le statut actuel du batch
  - 🩺 **Diagnostic** : Analyse les erreurs potentielles (OpenAI uniquement)
  - 🎯 **Traiter** : Récupère et intègre les résultats terminés

## Différences entre OpenAI et Anthropic

| Aspect | OpenAI Batch | Anthropic Batch |
|--------|--------------|-----------------|
| **Création** | Upload fichier JSONL | Envoi direct JSON |
| **Suivi** | Polling périodique | Polling périodique |
| **Résultats** | Téléchargement JSONL | Téléchargement JSONL via URL |
| **Diagnostic** | Outils avancés | Statut basique |
| **Coût** | 50% de réduction | 50% de réduction |

## Cycle de vie d'un batch Anthropic

1. **Création** (`launch_anthropic_batch`)
   - Préparation des requêtes au format Anthropic
   - Appel à l'API `POST /v1/messages/batches`
   - Retour de l'ID du batch

2. **Suivi** (`get_anthropic_batch_status`)
   - Appel à `GET /v1/messages/batches/{id}`
   - Statuts possibles : `in_progress`, `ended`, `failed`, `expired`, `cancelled`

3. **Récupération** (`get_anthropic_batch_results`)
   - Téléchargement depuis `results_url`
   - Parsing du fichier JSONL
   - Extraction du contenu généré

## Gestion des erreurs

### Clé API manquante
```
Configuration invalide pour le modèle claude-sonnet-4-20250514: 
Clé API manquante pour anthropic. Veuillez la configurer dans config/user.yaml 
ou comme variable d'environnement.
```

**Solution** : Ajoutez la clé API dans la configuration ou comme variable d'environnement.

### Modèle non supporté
Si vous utilisez un modèle non référencé, le système utilisera le fallback basé sur le nom :
- Modèles contenant "claude" → Anthropic
- Autres modèles → OpenAI

### Erreurs de batch
Les erreurs spécifiques aux batchs Anthropic sont affichées dans l'interface avec le statut détaillé.

## Architecture technique

### Nouveaux modules

- `src/core/anthropic_batch_processor.py` : Gestion complète de l'API Batch Anthropic
- Modifications dans `src/config_manager.py` : Support multi-fournisseur
- Modifications dans `src/core/process_tracker.py` : Suivi des fournisseurs
- Modifications dans `src/app.py` : Interface unifiée

### Fonctions clés

```python
# Lancement unifié (détection automatique du fournisseur)
process_id = launch_unified_batch_process(
    plan_items=sections,
    model="claude-sonnet-4-20250514",  # Détecté comme Anthropic
    config_manager=config_manager,
    # ... autres paramètres
)

# API Anthropic spécifique
batch_id = launch_anthropic_batch(prompts, model, api_key)
status = get_anthropic_batch_status(batch_id, api_key)
results = get_anthropic_batch_results(batch_id, api_key)
```

## Migration et compatibilité

### Processus existants
Les processus OpenAI existants continuent de fonctionner sans modification.

### Nouvelle configuration
Les anciens fichiers de configuration restent valides. Les nouveaux champs sont optionnels jusqu'à ce que vous utilisiez un modèle Anthropic.

### Tests
```bash
# Tests d'intégration
python -m pytest tests/test_anthropic_integration.py -v

# Tests de critères d'acceptation
python -m pytest tests/test_acceptance_criteria.py -v
```

## Avantages

1. **Économies** : 50% de réduction des coûts avec l'API Batch
2. **Transparence** : Sélection automatique du fournisseur
3. **Compatibilité** : Aucune régression sur l'existant
4. **Suivi unifié** : Interface commune pour tous les batchs
5. **Flexibilité** : Support facile de nouveaux modèles

## Support et dépannage

### Logs
Les logs détaillés sont disponibles dans la console pour diagnostiquer les problèmes.

### Tests de validation
Utilisez les tests d'acceptation pour valider votre configuration :
```bash
python -m pytest tests/test_acceptance_criteria.py::TestAcceptanceCriteria::test_criterion_6_missing_api_key_error -v
```

### Vérification de configuration
```python
from src.config_manager import ConfigManager

config = ConfigManager()
# Valide la configuration pour un modèle Anthropic
details = config.validate_model_config("claude-sonnet-4-20250514")
print(f"Modèle validé : {details}")
```
