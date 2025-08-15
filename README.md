
# Générateur d'Ouvrage Assisté par IA

Application Streamlit avancée pour la génération d'ouvrages à partir d'un plan DOCX et d'un corpus enrichi (Excel analysé). L'application utilise l'IA pour générer du contenu structuré et le raffiner automatiquement.

## 🚀 Fonctionnalités Principales

### ✨ Génération en Deux Étapes
- **IA 1 (Brouillon)** : Génère le contenu initial basé sur le corpus
- **IA 2 (Raffinement)** : Améliore et unifie le style du contenu généré

### 🤖 Modèles Supportés
- **OpenAI** : GPT-5, GPT-4.1, GPT-4.1-mini
- **Anthropic** : Claude 4 Sonnet, Claude 3.5 Sonnet

### 📊 Analyse Automatique
- Analyse automatique de la couverture du corpus par section
- Évaluation de la pertinence avec l'IA
- Métriques de performance en temps réel

### 💾 Gestion des Configurations
- Sauvegarde/chargement des configurations utilisateur
- Chargement automatique des fichiers depuis les chemins par défaut
- Variables d'environnement automatiques

## 🛠️ Installation

### Prérequis
- Python 3.8+
- pip

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Configuration
1. **Copiez le template d'environnement** :
   ```bash
   cp env.template .env
   ```

2. **Configurez vos clés API** dans le fichier `.env` :
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
   ```

3. **Configurez les chemins par défaut** dans `config/user.yaml` :
   ```yaml
   default_files:
     plan: "chemin/vers/plan_ouvrage.docx"
     corpus: "chemin/vers/corpus_ANALYZED.xlsx"
     keywords: "chemin/vers/keywords_mapping.json"
   ```

## 🚀 Lancement

### Méthode 1 : Script Python (Recommandé)
```bash
python run_app.py
```

### Méthode 2 : Streamlit Direct
```bash
streamlit run src/app.py
```

### Méthode 3 : Fichier Batch (Windows)
```bash
run_app.bat
```

L'application sera accessible à l'adresse : **http://localhost:8501**

## 📱 Interface Utilisateur

### Page 1 : Accueil & Fichiers
- Chargement des fichiers (plan DOCX, corpus Excel, mapping mots-clés)
- Chargement automatique depuis les chemins par défaut
- Détection automatique des clés API

### Page 2 : Configuration
- Configuration des modèles d'IA et paramètres
- Paramètres de génération avancés
- Styles DOCX et marges
- Sauvegarde/chargement des configurations

### Page 3 : Analyse & Préparation
- Couverture globale du corpus
- Analyse par section avec l'IA
- Mapping des colonnes du corpus

### Page 4 : Génération
- Génération manuelle ou automatique
- Traitement par section avec indicateurs de progression
- Export automatique en Markdown et DOCX

### Page 5 : Résultats & Export
- Consultation des brouillons et versions finales
- Raffinement automatique des sections
- Compilation du document complet
- Export final en Markdown et DOCX

## 🔧 Configuration Avancée

### Paramètres de Génération
- **Temperature** : Contrôle la créativité (0.0 = déterministe, 2.0 = très créatif)
- **Top-p** : Contrôle la diversité du vocabulaire
- **Tokens de sortie** : Limite la longueur du texte généré
- **Effort de raisonnement** : Spécifique aux modèles GPT-5
- **Verbosité** : Contrôle le niveau de détail

### Filtrage du Corpus
- **Score de pertinence** : Seuil minimum pour inclure une entrée
- **Citations maximum** : Limite le nombre de citations par section
- **Correspondances secondaires** : Inclut les correspondances de moindre qualité
- **Seuil de confiance** : Filtre par niveau de confiance

## 📁 Structure du Projet

```
Corpus ok/
├── src/                    # Code source principal
│   ├── app.py             # Application Streamlit
│   ├── config_manager.py  # Gestionnaire de configuration
│   └── core/              # Modules principaux
│       ├── corpus_manager.py    # Gestion du corpus
│       ├── prompt_builder.py    # Construction des prompts
│       └── utils.py             # Utilitaires
├── config/                 # Fichiers de configuration
│   ├── user.yaml          # Configuration utilisateur
│   ├── prompts.yaml       # Templates de prompts
│   └── saved/             # Configurations sauvegardées
├── tests/                  # Tests unitaires et d'intégration
├── output/                 # Fichiers générés
├── requirements.txt        # Dépendances Python
├── run_app.py             # Script de lancement principal
└── README.md              # Documentation
```

## 🧪 Tests

### Exécution des Tests
```bash
# Tous les tests
python -m pytest

# Tests avec détails
python -m pytest -v

# Tests spécifiques
python -m pytest tests/test_new_features.py -v
```

### Types de Tests
- **Tests unitaires** : Fonctions individuelles
- **Tests d'intégration** : Flux complets
- **Tests de performance** : Métriques et temps de réponse

## 🔍 Dépannage

### Problèmes Courants

#### Erreur d'Import
```bash
ModuleNotFoundError: No module named 'config_manager'
```
**Solution** : Utilisez `python run_app.py` au lieu de `streamlit run src/app.py`

#### Clés API Non Détectées
**Solution** : Vérifiez que le fichier `.env` est dans le bon répertoire et contient les bonnes clés

#### Fichiers Non Chargés
**Solution** : Vérifiez les chemins dans `config/user.yaml` et l'existence des fichiers

### Logs et Debug
- Les erreurs sont affichées dans l'interface Streamlit
- Vérifiez la console pour les messages de debug
- Les exports sont sauvegardés dans le dossier `output/`

## 🚀 Développement

### Ajout de Nouvelles Fonctionnalités
1. Modifiez le code source dans `src/`
2. Ajoutez les tests correspondants dans `tests/`
3. Mettez à jour la documentation
4. Testez avec `python -m pytest`

### Structure des Prompts
- Modifiez `config/prompts.yaml` pour changer les templates
- Utilisez les variables `{section_title}`, `{corpus}`, etc.
- Testez les prompts avec différents modèles

## 📄 Licence

Ce projet est développé pour un usage interne et éducatif.

## 🤝 Contribution

Pour contribuer au projet :
1. Créez une branche pour votre fonctionnalité
2. Ajoutez des tests pour les nouvelles fonctionnalités
3. Mettez à jour la documentation
4. Soumettez une pull request

---

**Version** : 2.0.0  
**Dernière mise à jour** : Décembre 2024  
**Développé avec** : Streamlit, OpenAI, Anthropic
