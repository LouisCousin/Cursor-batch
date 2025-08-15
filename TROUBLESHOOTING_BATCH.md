# Guide de Dépannage - Traitement par Lot (Batch API)

## 🚨 Erreurs Courantes et Solutions

### 1. "Aucun fichier de sortie disponible pour le batch"

**Cause :** Le batch est marqué comme "completed" mais OpenAI n'a pas généré de fichier de résultats.

**Solutions :**
1. **Utiliser le diagnostic** : Cliquez sur "🩺 Diagnostic" pour voir les détails
2. **Vérifier les erreurs** : Le diagnostic affichera les erreurs spécifiques
3. **Reprendre le processus** : Utilisez "🔄 Reprendre" pour relancer les sections en échec

**Actions automatiques :**
- Les sections sont automatiquement marquées en échec
- Un message d'erreur explicite est enregistré
- Le processus peut être repris facilement

### 2. Batch en statut "failed"

**Cause :** Échec côté OpenAI (problème de quota, contenu inapproprié, etc.)

**Solutions :**
1. **Diagnostic complet** : Utiliser le bouton "🩺 Diagnostic"
2. **Vérifier les quotas** : S'assurer d'avoir suffisamment de crédits API
3. **Contrôler le contenu** : Vérifier que les prompts respectent les conditions d'usage
4. **Reprendre** : Relancer les sections avec "🔄 Reprendre"

### 3. Batch bloqué en "in_progress" depuis longtemps

**Cause :** Le traitement peut prendre jusqu'à 24h selon OpenAI.

**Solutions :**
1. **Estimer le temps** : Utiliser "🔍 Vérifier" pour voir la progression
2. **Attendre** : Les batches peuvent être lents selon la charge OpenAI
3. **Vérifier après 24h** : Si toujours bloqué, contacter le support OpenAI

### 4. "ModuleNotFoundError: No module named 'stubs_batch'"

**Cause :** Problème d'import du module batch.

**Solutions :**
1. **Vérifier l'installation** : S'assurer que tous les fichiers sont présents
2. **Redémarrer l'application** : Relancer Streamlit
3. **Vérifier la structure** : Le fichier `stubs_batch.py` doit être à la racine

## 🛠️ Outils de Diagnostic

### Bouton "🔍 Vérifier"
- **Statut en temps réel** : État actuel du batch sur OpenAI
- **Progression** : Pourcentage d'avancement
- **Estimation** : Temps restant approximatif
- **Détails techniques** : Informations complètes de l'API

### Bouton "🩺 Diagnostic"
- **Problèmes détectés** : Liste des erreurs identifiées
- **Contenu d'erreur** : Détails spécifiques des échecs
- **Rapport complet** : Analyse technique complète
- **Recommandations** : Actions suggérées

### Bouton "🎯 Traiter les batches terminés"
- **Récupération automatique** : Traite tous les batches prêts
- **Gestion d'erreurs** : Marque les échecs appropriément
- **Export automatique** : Génère les fichiers de sortie

## 📊 Statuts de Batch

| Statut OpenAI | Description | Action |
|---------------|-------------|--------|
| `validating` | Validation du fichier d'entrée | Attendre |
| `in_progress` | Traitement en cours | Surveiller la progression |
| `finalizing` | Finalisation des résultats | Attendre |
| `completed` | Terminé avec succès | Récupérer les résultats |
| `failed` | Échec complet | Diagnostiquer et reprendre |
| `expired` | Expiré (>24h) | Reprendre le processus |
| `cancelled` | Annulé | Reprendre si nécessaire |

## 🔧 Actions de Récupération

### Reprise Automatique
- **Détection intelligente** : Identifie les sections à reprendre
- **Nouveau batch** : Crée automatiquement un nouveau lot
- **Historique préservé** : Garde la trace de toutes les tentatives

### Récupération Manuelle
1. **Identifier les échecs** : Page "Historique des Générations"
2. **Diagnostiquer** : Utiliser les outils de diagnostic
3. **Reprendre** : Cliquer sur "🔄 Reprendre"
4. **Surveiller** : Suivre le nouveau batch

## 💡 Conseils de Prévention

### Optimisation des Prompts
- **Taille raisonnable** : Éviter les prompts trop longs
- **Contenu approprié** : Respecter les conditions d'usage OpenAI
- **Format correct** : Vérifier la structure des données

### Gestion des Ressources
- **Quotas API** : Surveiller l'utilisation des crédits
- **Taille des lots** : Commencer par de petits batches pour tester
- **Timing** : Éviter les périodes de forte charge

### Surveillance
- **Vérifications régulières** : Utiliser les boutons d'actualisation
- **Notifications** : Noter les IDs de processus importants
- **Sauvegarde** : Télécharger les résultats immédiatement

## 📞 Support

### Logs d'Application
- Les erreurs sont enregistrées dans les logs Python
- Utiliser le diagnostic pour plus de détails
- Conserver les IDs de batch pour le support

### Contact OpenAI
- Pour les problèmes de quota ou de service
- Fournir l'ID du batch problématique
- Inclure les détails du diagnostic

### Documentation Technique
- [API Batch OpenAI](https://platform.openai.com/docs/guides/batch)
- [Limites et quotas](https://platform.openai.com/docs/guides/rate-limits)
- [Support OpenAI](https://help.openai.com/)

---

*Ce guide couvre les problèmes les plus courants. Pour des cas spécifiques, utilisez les outils de diagnostic intégrés.*
