# Guide d'Utilisation - Traitement par Lot (Batch API OpenAI)

## 📋 Présentation

Le traitement par lot permet de générer des contenus de manière asynchrone via l'API Batch d'OpenAI, offrant des avantages significatifs pour les traitements massifs :

- **Réduction des coûts** : Tarifs réduits de 50% par rapport à l'API standard
- **Robustesse** : Pas de risque de timeout ou d'interruption de connexion
- **Flexibilité** : Traitement en arrière-plan, libère l'interface utilisateur

## 🚀 Comment Utiliser

### 1. Lancement d'un Processus Batch

1. **Accéder à la page "4. Génération"**
2. **Sélectionner le mode de génération** :
   - Manuel (une section) ou Automatique (plusieurs sections)
3. **Choisir le type de traitement** :
   - **Synchrone** : Génération immédiate (mode traditionnel)
   - **Batch** : Traitement différé via l'API Batch OpenAI
4. **Sélectionner les sections** à traiter
5. **Cliquer sur "🚀 Lancer la Génération"**

### 2. Suivi des Processus

1. **Accéder à la page "6. Historique des Générations"**
2. **Identifier les processus de type "batch"**
3. **Utiliser les boutons d'action** :
   - **🔄 Actualiser Statut** : Met à jour le statut depuis OpenAI
   - **🔄 Reprendre** : Relance les sections en échec
   - **🗑️ Supprimer** : Supprime le processus de l'historique

### 3. Récupération des Résultats

#### Automatique
- Utilisez le bouton **"🎯 Traiter les batches terminés"** dans la page d'historique
- L'application récupère automatiquement tous les résultats disponibles

#### Manuel
- Cliquez sur **"🔍 Vérifier"** sur un lot spécifique
- Consultez la progression et les estimations de temps

## 📊 Fonctionnalités Avancées

### Suivi en Temps Réel

- **Progression en pourcentage** : Affichage du taux de complétion
- **Estimation de temps** : Calcul du temps restant estimé
- **Détails techniques** : Accès aux informations complètes de l'API

### Reprise des Processus

- **Détection automatique** des sections en échec
- **Relance intelligente** des tâches non terminées
- **Historique complet** des tentatives

### Export Automatique

- **Markdown (.md)** : Format texte avec métadonnées
- **Word (.docx)** : Document formaté avec styles
- **Traçabilité** : Chaque fichier contient l'ID du batch et l'horodatage

## ⚙️ Configuration Requise

### Clés API
- **Clé API OpenAI** : Obligatoire pour le traitement batch
- Configuration dans la page "2. Configuration"

### Modèles Supportés
- Tous les modèles OpenAI compatibles avec l'API Batch
- Configuration via le sélecteur de modèle "Brouillon"

### Paramètres de Corpus
- Les mêmes paramètres que le mode synchrone
- Configuration dans la page "2. Configuration" → "Paramètres de Filtrage du Corpus"

## 🔧 Dépannage

### Problèmes Courants

#### "Erreur lors du lancement du processus par lot"
- **Vérifier la clé API OpenAI** dans la configuration
- **S'assurer que le corpus est chargé** avec des données pertinentes
- **Vérifier la connexion internet**

#### "Aucune section à reprendre"
- Le processus est déjà terminé avec succès
- Toutes les sections ont été traitées correctement

#### "Batch en statut 'failed'"
- Le lot a échoué côté OpenAI
- Utiliser le bouton "🔄 Reprendre" pour relancer les sections
- Vérifier les quotas API OpenAI

### Limitations

- **OpenAI uniquement** : Le traitement batch n'est pas disponible pour Anthropic
- **Délai de traitement** : Les batches peuvent prendre jusqu'à 24h
- **Taille des fichiers** : Limitation des fichiers d'entrée selon l'API OpenAI

## 📈 Bonnes Pratiques

### Quand Utiliser le Mode Batch

✅ **Recommandé pour :**
- Génération de plus de 5 sections
- Traitement de documents volumineux
- Travail sur des projets non urgents
- Optimisation des coûts

❌ **Non recommandé pour :**
- Travail itératif rapide
- Sections uniques
- Prototypage et tests
- Besoins immédiats

### Optimisation

1. **Grouper les sections** : Traiter plusieurs sections ensemble
2. **Vérifier régulièrement** : Utiliser l'actualisation automatique
3. **Surveiller les quotas** : Respecter les limites de l'API OpenAI
4. **Sauvegarder les résultats** : Télécharger les fichiers générés

## 🛠️ Support Technique

### Logs et Diagnostics

- Les erreurs sont enregistrées dans les logs de l'application
- Utiliser les boutons "🔍 Vérifier" pour diagnostiquer les problèmes
- Consulter la section "Détails techniques" pour plus d'informations

### Historique des Processus

- **Conservation** : Tous les processus sont sauvegardés
- **Nettoyage** : Utiliser l'option de nettoyage pour les anciens processus
- **Export** : Possibilité d'exporter l'historique complet

---

*Ce guide couvre les fonctionnalités principales du traitement par lot. Pour des questions spécifiques, consultez la documentation de l'API OpenAI Batch.*
