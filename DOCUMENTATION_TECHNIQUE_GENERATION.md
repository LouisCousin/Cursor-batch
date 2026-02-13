# Documentation Technique : Processus de Génération de Texte

## Vue d'ensemble

Ce document détaille le pipeline complet de génération de texte utilisé dans le **Générateur d'Ouvrage Assisté par IA**. Le système utilise une approche en **deux étapes** avec orchestration intelligente pour générer des ouvrages structurés à partir d'un corpus enrichi.

---

## Architecture Générale

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE DE PRÉPARATION                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Chargement des fichiers                                     │
│     ├── Plan DOCX (structure de l'ouvrage)                      │
│     ├── Corpus Excel (données enrichies avec scores)            │
│     └── Mapping JSON (correspondances mots-clés)                │
│                                                                  │
│  2. Analyse et filtrage du corpus                               │
│     ├── CorpusManager : filtrage par section                    │
│     ├── Scoring de pertinence                                   │
│     └── Extraction des citations pertinentes                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                PHASE D'ORCHESTRATION                             │
├─────────────────────────────────────────────────────────────────┤
│  3. Création des tâches de génération                           │
│     ├── GenerationOrchestrator : gestion des dépendances        │
│     ├── GenerationTask : représentation d'une section           │
│     └── OrchestrationContext : partage des résumés              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               PHASE DE GÉNÉRATION (pour chaque section)          │
├─────────────────────────────────────────────────────────────────┤
│  4. Construction du prompt (PromptBuilder)                      │
│     ├── Intégration du contexte des sections précédentes        │
│     ├── Formatage du corpus filtré                              │
│     ├── Injection des statistiques                              │
│     └── Application du template de prompt                       │
│                                                                  │
│  5. ÉTAPE 1 : Génération du brouillon (IA 1)                    │
│     ├── Appel API (OpenAI ou Anthropic)                         │
│     ├── Gestion des retry automatiques                          │
│     ├── Extraction du texte brut                                │
│     └── Création d'un résumé flash (200-500 tokens)             │
│                                                                  │
│  6. ÉTAPE 2 : Raffinement du texte (IA 2)                       │
│     ├── Construction du prompt de raffinement                   │
│     ├── Appel API avec consignes de style                       │
│     ├── Harmonisation du ton et de la terminologie              │
│     └── Préservation des citations APA                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE D'EXPORT                                  │
├─────────────────────────────────────────────────────────────────┤
│  7. Génération des fichiers de sortie                           │
│     ├── Export Markdown (.md)                                   │
│     ├── Export DOCX stylé                                       │
│     └── Génération de la bibliographie APA                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Détail Technique des Étapes

### 1. Préparation des Données

#### 1.1 Chargement du Plan DOCX
**Fichier** : `src/core/utils.py:138-154`

```python
def parse_docx_plan(docx_path: str) -> List[Dict[str, Any]]
```

**Processus** :
1. Lecture du document Word avec `python-docx`
2. Détection des styles de titre (Heading 1-3, Titre 1-3)
3. Numérotation automatique hiérarchique (1, 1.1, 1.1.1)
4. Extraction du code et titre de chaque section

**Sortie** : Liste de dictionnaires `{"code": "1.2", "title": "Titre section", "level": 2}`

#### 1.2 Gestion du Corpus
**Fichier** : `src/core/corpus_manager.py`

Le `CorpusManager` gère le filtrage intelligent du corpus :

**Fonctionnalités** :
- **Filtrage par pertinence** : Sélection des entrées selon un score minimum
- **Limitation du nombre** : Maximum de citations par section
- **Filtrage par confiance** : Exclusion des correspondances de faible qualité
- **Support des colonnes personnalisées** : Détection automatique des colonnes textuelles

**Colonnes détectées automatiquement** :
- Texte : `Texte`, `Extrait`, `Citation`, `Content`, `Text`
- Métadonnées : `Score`, `MatchType`, `Confidence`

---

### 2. Orchestration des Tâches

#### 2.1 Architecture d'Orchestration
**Fichier** : `src/core/orchestrator.py`

**Classes principales** :

##### `GenerationTask` (ligne 26-44)
Représente une tâche de génération pour une section.

```python
@dataclass
class GenerationTask:
    id: str                        # Identifiant unique
    section_code: str              # Code de section (ex: "1.2")
    section_title: str             # Titre de la section
    dependencies: List[str]        # IDs des sections dont elle dépend
    status: TaskStatus             # EN_ATTENTE, PRET, EN_COURS, TERMINE, ECHEC
    result_text: Optional[str]     # Texte généré
    summary: Optional[str]         # Résumé flash pour les sections suivantes
    error_message: Optional[str]   # Message d'erreur si échec
    start_time: Optional[datetime] # Heure de début
    end_time: Optional[datetime]   # Heure de fin
```

##### `OrchestrationContext` (ligne 48-82)
Contexte partagé thread-safe entre les tâches.

**Méthodes clés** :
- `add_summary(task_id, summary)` : Ajoute un résumé de manière thread-safe
- `get_context_for_task(task)` : Construit le contexte textuel à partir des résumés précédents

**Exemple de contexte généré** :
```
--- CONTEXTE DES SECTIONS PRÉCÉDENTES ---
Résumé de la section précédente (1.1_Introduction):
Cette section présente les concepts fondamentaux...

Résumé de la section précédente (1.2_Méthode):
L'approche méthodologique s'appuie sur...
--- FIN DU CONTEXTE ---
```

##### `GenerationOrchestrator` (ligne 85-391)
Gestionnaire principal de l'exécution.

**Paramètres d'initialisation** :
- `tasks` : Liste des tâches à exécuter
- `progress_callback` : Fonction appelée à chaque changement de statut
- `max_workers` : Nombre de workers parallèles (limité à 4 pour les API)

**Méthodes principales** :

1. **`_get_ready_tasks()`** (ligne 141-159)
   - Identifie les tâches prêtes à être exécutées
   - Vérifie que toutes les dépendances sont terminées
   - Change le statut de `EN_ATTENTE` à `PRET`

2. **`_execute_task(task)`** (ligne 176-225)
   - Construit le contexte pour la tâche
   - Appelle la fonction de génération
   - Extrait ou génère un résumé
   - Met à jour le contexte partagé
   - Gère les erreurs

3. **`run()`** (ligne 227-288) - **Mode synchrone (Streamlit)**
   - Traite les tâches une par une
   - Compatible avec Streamlit
   - Appelle le callback de progression après chaque tâche

4. **`run_parallel()`** (ligne 290-358) - **Mode parallèle**
   - Utilise `ThreadPoolExecutor`
   - Exécute jusqu'à 4 tâches simultanément
   - Peut causer des warnings Streamlit

#### 2.2 Gestion des Dépendances

**Fonction** : `create_linear_dependency_tasks()` (ligne 394-437)

Crée des dépendances linéaires : chaque section dépend de la précédente.

**Exemple** :
```
Section 1.1 → aucune dépendance (PRET immédiatement)
Section 1.2 → dépend de 1.1 (EN_ATTENTE jusqu'à completion de 1.1)
Section 1.3 → dépend de 1.2 (EN_ATTENTE jusqu'à completion de 1.2)
```

---

### 3. Construction des Prompts

#### 3.1 PromptBuilder
**Fichier** : `src/core/prompt_builder.py`

##### Méthode `build_draft_prompt()` (ligne 11-81)

**Entrées** :
- `section_title` : Titre de la section à générer
- `corpus_df` : DataFrame filtré du corpus
- `keywords` : Liste des mots-clés détectés
- `previous_summaries` : Résumés des sections précédentes
- `stats` : Statistiques (nombre d'entrées, score moyen)

**Processus de construction** :

1. **Formatage du corpus** (lignes 28-52)
   ```python
   for _, row in corpus_df.iterrows():
       text = row['Texte']  # ou 'Extrait', 'Citation', etc.
       metadata = []
       if 'Score' in row:
           metadata.append(f"Score: {row['Score']:.2f}")
       if 'MatchType' in row:
           metadata.append(f"Type: {row['MatchType']}")
       if 'Confidence' in row:
           metadata.append(f"Confiance: {row['Confidence']}%")

       corpus_entries.append(f"- {text} ({', '.join(metadata)})")
   ```

2. **Calcul des statistiques** (lignes 59-69)
   - Nombre d'entrées du corpus
   - Score moyen de pertinence
   - Liste des mots-clés

3. **Application du template** (lignes 72-79)
   ```python
   prompt = template
   prompt = prompt.replace("{section_title}", section_title)
   prompt = prompt.replace("{corpus}", corpus_text)
   prompt = prompt.replace("{keywords_found}", keywords)
   prompt = prompt.replace("{corpus_count}", str(count))
   prompt = prompt.replace("{avg_score}", str(avg_score))
   prompt = prompt.replace("{previous_summaries}", summaries)
   ```

**Template utilisé** (fichier `config/prompts.yaml:2-18`) :

```yaml
drafter: |
  # Rédaction de la sous-partie {section_title}
  ## Contexte et objectifs
  {section_plan}
  ## Consignes de rédaction
  - Utilise un maximum d'analyses et citations du corpus fourni.
  - Intègre chaque citation entre guillemets et au format APA : (Auteur, Année).
  - Structure en markdown.
  - Termine par un résumé flash de 200-500 tokens.
  ## Statistiques du corpus
  - Nombre d'éléments: {corpus_count}
  - Score moyen: {avg_score}
  - Mots-clés détectés: {keywords_found}
  ## Corpus à utiliser
  {corpus}
  ## Résumés flash précédents
  {previous_summaries}
```

##### Méthode `build_refine_prompt()` (ligne 83-99)

**Entrées** :
- `draft_markdown` : Texte du brouillon à raffiner
- `style_guidelines` : Consignes de style spécifiques (optionnel)

**Template utilisé** (fichier `config/prompts.yaml:20-22`) :

```yaml
refiner: |
  Tu es un éditeur de style. Réécris et condense le texte fourni en assurant cohérence,
  clarté et fluidité, en gardant les citations APA. Harmonise le ton et la terminologie.
```

---

### 4. Génération du Brouillon (IA 1)

#### 4.1 Appel à l'API OpenAI
**Fichier** : `src/core/utils.py:47-106`

```python
@retry_on_failure
def call_openai(
    model_name: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_output_tokens: int = 1024,
    reasoning_effort: str = "medium",  # GPT-5 uniquement
    verbosity: str = "medium"           # GPT-5 uniquement
) -> str
```

**Détection du modèle** (ligne 61) :
```python
is_gpt5 = "gpt-5" in model_name.lower()
```

**API pour GPT-5** (lignes 64-70) :
```python
response = client.responses.create(
    model=model_name,
    input=prompt,
    reasoning={"effort": reasoning_effort},  # "low", "medium", "high"
    text={"verbosity": verbosity}           # "low", "medium", "high"
)
```

**API pour GPT-4 et antérieurs** (lignes 73-79) :
```python
response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,  # 0.0 = déterministe, 2.0 = très créatif
    top_p=top_p,             # 0.0-1.0, contrôle la diversité
    max_tokens=max_output_tokens
)
```

**Extraction de la réponse** (lignes 82-106) :

Pour GPT-5 :
```python
for item in response.output:
    if item.type == "message":
        for content_item in item.content:
            if content_item.type == "output_text":
                return content_item.text
```

Pour GPT-4 et antérieurs :
```python
return response.choices[0].message.content
```

#### 4.2 Appel à l'API Anthropic (Claude)
**Fichier** : `src/core/utils.py:108-136`

```python
@retry_on_failure
def call_anthropic(
    model_name: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_output_tokens: int = 1024
) -> str
```

**Appel API** (lignes 118-125) :
```python
client = anthropic.Anthropic(api_key=api_key)
msg = client.messages.create(
    model=model_name,
    max_tokens=max_output_tokens,
    temperature=temperature,
    top_p=top_p,
    messages=[{"role": "user", "content": prompt}]
)
```

**Extraction du texte** (lignes 127-136) :
```python
parts = []
for blk in msg.content:
    if isinstance(blk, dict) and blk.get("type") == "text":
        parts.append(blk.get("text", ""))
    else:
        parts.append(getattr(blk, "text", ""))
return "\n".join(parts).strip()
```

#### 4.3 Mécanisme de Retry
**Fichier** : `src/core/utils.py:15-27`

**Décorateur** : `@retry_on_failure`

```python
def retry_on_failure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = len(API_RETRY_DELAYS) + 1  # Défini dans config_manager
        for i in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i < len(API_RETRY_DELAYS):
                    time.sleep(API_RETRY_DELAYS[i])  # [2, 5, 10] par défaut
                else:
                    raise  # Lève l'erreur après le dernier essai
    return wrapper
```

**Configuration des délais** (fichier `config_manager.py`) :
```python
API_RETRY_DELAYS = [2, 5, 10]  # Délais en secondes entre les tentatives
```

**Exemple de comportement** :
1. Tentative 1 → Échec → Attente 2s
2. Tentative 2 → Échec → Attente 5s
3. Tentative 3 → Échec → Attente 10s
4. Tentative 4 → Échec → Exception levée

#### 4.4 Gestion des Tokens
**Fichier** : `src/core/utils.py:29-44`

```python
def calculate_max_input_tokens(model_name: str, requested_output_tokens: int) -> int
```

**Formule** :
```
max_input = context_total - output_tokens - marge_sécurité

où :
  marge_sécurité = max(200, 10% × output_tokens)
```

**Exemple** :
- Modèle : GPT-4 (context = 128 000 tokens)
- Output demandé : 4 000 tokens
- Marge : max(200, 400) = 400 tokens
- **Input max : 128 000 - 4 000 - 400 = 123 600 tokens**

**Troncature du prompt** :
```python
def truncate_to_tokens(text: str, max_tokens: int, model: str) -> str
```

Utilise `tiktoken` si disponible, sinon heuristique : **1 token ≈ 4 caractères**

---

### 5. Raffinement du Texte (IA 2)

#### 5.1 Processus de Raffinement

**Objectifs** :
1. **Harmonisation du style** : Ton uniforme dans tout l'ouvrage
2. **Condensation** : Élimination des redondances
3. **Clarification** : Amélioration de la fluidité
4. **Préservation** : Maintien des citations APA

**Prompt de raffinement** (généré par `build_refine_prompt()`) :

```
Tu es un éditeur de style. Réécris et condense le texte fourni en assurant cohérence,
clarté et fluidité, en gardant les citations APA. Harmonise le ton et la terminologie.

[Consignes de style spécifiques optionnelles]

---

Texte à raffiner:
[BROUILLON COMPLET]
```

#### 5.2 Appel API pour le Raffinement

**Même mécanisme que la génération** mais avec :
- **Temperature plus basse** (0.5-0.7) pour moins de créativité
- **Prompt axé sur l'édition** plutôt que la création
- **Conservation stricte des citations**

---

### 6. Extraction du Résumé Flash

#### 6.1 Méthode Automatique
**Fichier** : `src/core/orchestrator.py:119-139`

```python
def _extract_summary(self, text: str) -> str
```

**Algorithme** :
1. Diviser le texte en paragraphes
2. Si ≤2 paragraphes : prendre les 500 premiers caractères
3. Sinon : concaténer premier et dernier paragraphe avec `[...]`
4. Limiter à 800 caractères

**Exemple** :
```python
paragraphs = ["Introduction...", "Développement...", "Conclusion..."]
summary = paragraphs[0] + " [...] " + paragraphs[-1]
# "Introduction... [...] Conclusion..."
```

#### 6.2 Résumé Fourni par l'IA

Si l'IA génère un résumé flash en fin de texte (comme demandé dans le prompt) :
```python
task.summary = summary or self._extract_summary(text)
```

Le résumé de l'IA est prioritaire, sinon extraction automatique.

---

### 7. Export des Résultats

#### 7.1 Export Markdown
**Fichier** : `src/core/utils.py:216-223`

```python
def export_markdown(text_md: str, base_name: str, mode: str, export_dir: str) -> str
```

**Processus** :
1. Création du dossier `output/` si inexistant
2. Génération du nom : `YYYYMMDD-HHMMSS_mode_basename.md`
3. Écriture du fichier en UTF-8

**Exemple** :
```
output/20250213-143022_draft_Introduction.md
output/20250213-143542_final_Introduction.md
```

#### 7.2 Export DOCX Stylé
**Fichier** : `src/core/utils.py:225-232` et `169-188`

```python
def export_docx(text_md: str, base_name: str, mode: str, export_dir: str, styles: dict) -> str
def generate_styled_docx(markdown_text: str, output_path: str, styles: Dict[str, Any]) -> None
```

**Processus de conversion Markdown → DOCX** :

1. **Création du document** (ligne 174)
   ```python
   doc = Document()
   ```

2. **Application des marges** (lignes 157-161)
   ```python
   section.top_margin = Cm(styles.get("margin_top", 2.5))
   section.bottom_margin = Cm(styles.get("margin_bottom", 2.5))
   section.left_margin = Cm(styles.get("margin_left", 2.5))
   section.right_margin = Cm(styles.get("margin_right", 2.5))
   ```

3. **Parsing du Markdown** (lignes 176-187)
   ```python
   for line in markdown_text.splitlines():
       if line.startswith("# "):      # Titre niveau 1
           _add_paragraph(doc, line[2:], h1_size, font_family)
       elif line.startswith("## "):   # Titre niveau 2
           _add_paragraph(doc, line[3:], h2_size, font_family)
       elif line.startswith("- "):    # Liste à puces
           p = doc.add_paragraph(line[2:])
           p.style = "List Bullet"
       else:                          # Paragraphe normal
           _add_paragraph(doc, line, body_size, font_family)
   ```

4. **Sauvegarde** (ligne 188)
   ```python
   doc.save(output_path)
   ```

**Styles par défaut** (fichier `config_manager.py`) :
```python
DEFAULT_STYLES = {
    "font_family": "Calibri",
    "font_size_body": 11,
    "font_size_h1": 16,
    "font_size_h2": 14,
    "margin_top": 2.5,
    "margin_bottom": 2.5,
    "margin_left": 2.5,
    "margin_right": 2.5
}
```

#### 7.3 Génération de la Bibliographie
**Fichier** : `src/core/utils.py:190-192` et `234-257`

```python
def extract_used_references_apa(text_md: str) -> List[str]
def generate_bibliography(used: List[str], excel_path: str) -> str
```

**Étape 1 : Extraction des citations** (lignes 190-192)
```python
# Regex pour détecter (Auteur, Année)
patterns = re.findall(r"\(([^,]+),\s*(\d{4})\)", text_md)
# Retourne ["Dupont, 2020", "Martin, 2021", ...]
```

**Étape 2 : Récupération des références complètes** (lignes 234-257)
1. Lecture du fichier Excel (feuille "Bibliographie")
2. Détection des colonnes :
   - `Référence APA complète` ou `reference apa complete`
   - `Référence courte` ou `reference courte`
3. Correspondance avec les citations utilisées
4. Formatage en liste Markdown

**Exemple de sortie** :
```markdown
- Dupont, J. (2020). Titre de l'article. Revue Scientifique, 12(3), 45-67.
- Martin, A., & Lefebvre, B. (2021). Ouvrage de référence. Paris: Éditions X.
```

---

## Paramètres de Configuration

### Modèles Disponibles

**OpenAI** (fichier `config_manager.py`) :
```python
AVAILABLE_OPENAI_MODELS = [
    "gpt-5",           # Nouveau modèle avec reasoning_effort
    "gpt-4.1",
    "gpt-4.1-mini"
]
```

**Anthropic** :
```python
AVAILABLE_ANTHROPIC_MODELS = [
    "claude-4-sonnet-20250514",
    "claude-3-5-sonnet-20241022"
]
```

### Limites de Contexte

```python
MODEL_LIMITS = {
    "gpt-5": {"context": 200000, "output": 100000},
    "gpt-4.1": {"context": 128000, "output": 16384},
    "gpt-4.1-mini": {"context": 128000, "output": 16384},
    "claude-4-sonnet-20250514": {"context": 200000, "output": 8192},
    "claude-3-5-sonnet-20241022": {"context": 200000, "output": 8192}
}
```

### Paramètres de Génération

**Temperature** (0.0 - 2.0) :
- `0.0` : Complètement déterministe
- `0.7` : Équilibré (par défaut)
- `1.5` : Très créatif
- `2.0` : Maximum de créativité

**Top-p** (0.0 - 1.0) :
- `0.1` : Vocabulaire très restreint
- `0.9` : Équilibré (par défaut)
- `1.0` : Vocabulaire complet

**Reasoning Effort** (GPT-5 uniquement) :
- `low` : Raisonnement rapide
- `medium` : Équilibré (par défaut)
- `high` : Raisonnement approfondi

**Verbosity** (GPT-5 uniquement) :
- `low` : Réponses concises
- `medium` : Équilibré (par défaut)
- `high` : Réponses détaillées

---

## Gestion des Erreurs et Logs

### Statuts des Tâches

```python
class TaskStatus(Enum):
    EN_ATTENTE = "EN_ATTENTE"  # Attend que les dépendances se terminent
    PRET = "PRÊT"              # Prête à être exécutée
    EN_COURS = "EN_COURS"      # En cours d'exécution
    TERMINE = "TERMINÉ"        # Terminée avec succès
    ECHEC = "ÉCHEC"            # Terminée en erreur
```

### Messages d'Erreur

Chaque `GenerationTask` stocke :
- `error_message` : Description de l'erreur
- `start_time` / `end_time` : Horodatage pour diagnostics

**Exemple** :
```python
task.status = TaskStatus.ECHEC
task.error_message = "Erreur API: Rate limit exceeded (429)"
```

### Statistiques d'Exécution

```python
orchestrator.get_statistics()
```

**Retourne** :
```python
{
    'total_tasks': 10,
    'completed': 8,
    'failed': 2,
    'in_progress': 0,
    'waiting': 0,
    'completion_rate': 80.0,  # Pourcentage
    'total_execution_time': 245.3  # Secondes
}
```

---

## Optimisations et Performances

### 1. Gestion du Parallélisme

**Mode synchrone** (`run()`) :
- Traite les tâches une par une
- Sûr pour Streamlit
- Plus lent mais sans warnings

**Mode parallèle** (`run_parallel()`) :
- Utilise `ThreadPoolExecutor`
- Max 4 workers simultanés (limite API)
- Plus rapide mais peut causer des warnings Streamlit

### 2. Cache et Réutilisation

**Résumés flash** :
- Stockés dans `OrchestrationContext.summaries`
- Partagés entre tâches via thread-safe lock
- Évite la régénération de contexte

**Corpus pré-filtré** :
- Filtrage une seule fois par `CorpusManager`
- Réutilisé pour toutes les sections

### 3. Limitation de Tokens

**Troncature intelligente** :
- Utilisation de `tiktoken` pour comptage précis
- Fallback heuristique (4 chars/token)
- Préservation de l'intégrité du prompt

---

## Exemples de Flux Complets

### Exemple 1 : Génération d'une Section Simple

**Entrées** :
- Section : `1.2 - Méthodologie`
- Corpus : 15 citations (score moyen : 8.2/10)
- Mots-clés : `["recherche qualitative", "entretiens", "analyse thématique"]`

**Étapes** :

1. **Construction du prompt** :
   ```
   # Rédaction de la sous-partie Méthodologie
   ## Statistiques du corpus
   - Nombre d'éléments: 15
   - Score moyen: 8.20
   - Mots-clés détectés: recherche qualitative, entretiens, analyse thématique
   ## Corpus à utiliser
   - "L'approche qualitative permet..." (Score: 9.20, Confiance: 95%)
   - "Les entretiens semi-directifs..." (Score: 8.50, Confiance: 90%)
   [...]
   ```

2. **Appel API (GPT-4.1)** :
   - Temperature: 0.7
   - Max tokens: 4000
   - Durée: ~12 secondes

3. **Résultat brouillon** :
   ```markdown
   ## Méthodologie

   L'approche adoptée repose sur une recherche qualitative...
   Selon Dupont (2020), "L'approche qualitative permet..."
   [...]

   **Résumé flash**: Cette section présente la méthodologie...
   ```

4. **Raffinement (Claude 3.5 Sonnet)** :
   - Temperature: 0.6
   - Max tokens: 4000
   - Durée: ~10 secondes

5. **Résultat final** :
   ```markdown
   ## Méthodologie

   Notre approche méthodologique s'appuie sur...
   Comme l'affirme Dupont (2020), "L'approche qualitative..."
   [...]
   ```

6. **Export** :
   - `output/20250213-143542_draft_Methodologie.md`
   - `output/20250213-143612_final_Methodologie.md`
   - `output/20250213-143612_final_Methodologie.docx`

**Temps total** : ~25 secondes

### Exemple 2 : Génération avec Dépendances

**Plan** :
```
1.1 - Introduction générale
1.2 - Contexte historique
1.3 - Problématique
```

**Ordre d'exécution** :

```
[t=0s]   Task 1.1: EN_ATTENTE → PRET (aucune dépendance)
[t=1s]   Task 1.1: PRET → EN_COURS
[t=18s]  Task 1.1: EN_COURS → TERMINE
         → Résumé stocké: "Cette introduction pose les bases..."
         → Task 1.2: EN_ATTENTE → PRET (dépendance satisfaite)

[t=19s]  Task 1.2: PRET → EN_COURS
         → Contexte ajouté: "Résumé de 1.1: Cette introduction..."
[t=35s]  Task 1.2: EN_COURS → TERMINE
         → Résumé stocké: "Le contexte historique montre..."
         → Task 1.3: EN_ATTENTE → PRET

[t=36s]  Task 1.3: PRET → EN_COURS
         → Contexte: "Résumé de 1.2: Le contexte historique..."
[t=52s]  Task 1.3: EN_COURS → TERMINE

Total: 52 secondes pour 3 sections
```

---

## Fichiers de Configuration

### `config/prompts.yaml`

Contient les templates de prompts :
- `drafter` : Template pour la génération du brouillon
- `refiner` : Template pour le raffinement

**Variables disponibles** :
- `{section_title}` : Titre de la section
- `{section_plan}` : Plan de la section
- `{corpus}` : Corpus formaté
- `{keywords_found}` : Mots-clés détectés
- `{corpus_count}` : Nombre d'entrées
- `{avg_score}` : Score moyen
- `{previous_summaries}` : Résumés des sections précédentes

### `config/user.yaml`

Configuration utilisateur :
```yaml
default_files:
  plan: "chemin/vers/plan_ouvrage.docx"
  corpus: "chemin/vers/corpus_ANALYZED.xlsx"
  keywords: "chemin/vers/keywords_mapping.json"

default_models:
  draft: "gpt-4.1"
  refine: "claude-3-5-sonnet-20241022"

default_params:
  temperature: 0.7
  top_p: 0.9
  max_output_tokens: 4000
```

---

## Conclusion

Ce système de génération de texte utilise une architecture sophistiquée combinant :

1. **Orchestration intelligente** avec gestion des dépendances
2. **Prompts contextuels** enrichis par les sections précédentes
3. **Approche en deux étapes** (brouillon + raffinement)
4. **Support multi-modèles** (OpenAI et Anthropic)
5. **Gestion robuste des erreurs** avec retry automatique
6. **Export professionnel** (Markdown et DOCX stylé)

Le résultat est un système capable de générer des ouvrages complets, cohérents et de qualité professionnelle à partir d'un corpus enrichi.

---

**Fichiers sources principaux** :
- `src/core/orchestrator.py` : Orchestration et gestion des tâches
- `src/core/prompt_builder.py` : Construction des prompts
- `src/core/utils.py` : Appels API et utilitaires
- `src/core/corpus_manager.py` : Gestion du corpus
- `src/app.py` : Interface Streamlit
- `config/prompts.yaml` : Templates de prompts
