
from __future__ import annotations
import os, copy, yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path

__version__ = "2.0.0"

# Chargement automatique des variables d'environnement
def load_env_vars():
    """Charge les variables d'environnement depuis un fichier .env s'il existe."""
    env_dir = os.getenv("env_dir")
    env_file = Path(env_dir) / ".env" if env_dir else Path(".env")
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"Warning: Impossible de charger le fichier .env: {e}")

# Charger les variables d'environnement au démarrage
load_env_vars()

# Catalogues modèles
AVAILABLE_DRAFTER_MODELS = ["GPT-4.1", "GPT-4.1 mini", "GPT-4.1 nano"]
AVAILABLE_REFINER_MODELS = ["Claude 4 Sonnet"]

# Fournisseurs détaillés
AVAILABLE_OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini"]
AVAILABLE_ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3.5-sonnet-20240620"]

MODEL_ALIASES = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "Claude 4 Sonnet": "claude-3.5-sonnet-20240620",
}

# Limites des modèles : source unique de vérité pour les capacités
MODEL_LIMITS = {
    # Modèle: {"context": total_tokens, "max_output": max_tokens_sortie}
    "gpt-5": {"context": 400000, "max_output": 128000},
    "gpt-5-mini": {"context": 400000, "max_output": 128000},
    "gpt-5-nano": {"context": 400000, "max_output": 128000},
    "gpt-4.1": {"context": 1047576, "max_output": 32768},
    "gpt-4.1-mini": {"context": 1000000, "max_output": 32768},
    "claude-sonnet-4-20250514": {"context": 200000, "max_output": 64000},
    "claude-3.5-sonnet-20240620": {"context": 200000, "max_output": 8192},
}

# Modèles disponibles mis à jour
AVAILABLE_OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini"]
AVAILABLE_ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3.5-sonnet-20240620"]

# Defaults
DEFAULT_DRAFTER_MODEL = "GPT-4.1"
DEFAULT_REFINER_MODEL = "Claude 4 Sonnet"
API_RETRY_DELAYS = [60, 300]

DEFAULT_STYLES = {
    "font_family": "Calibri",
    "font_size_body": 11,
    "font_size_h1": 18,
    "font_size_h2": 14,
    "margin_top": 2.5,
    "margin_bottom": 2.5,
    "margin_left": 2.5,
    "margin_right": 2.5,
}

# Chemins par défaut (peuvent être surchargés dans config/user.yaml)
DEFAULT_PATHS = {
    "plan_docx": "",
    "excel_corpus": "",
    "keywords_json": "",
    "env_dir": "",
}

# Corpus filtering
MIN_RELEVANCE_SCORE = 7
MAX_CITATIONS_PER_SECTION = 30
INCLUDE_SECONDARY_MATCHES = True
CONFIDENCE_THRESHOLD = 60

DEFAULT_GPT_PROMPT_TEMPLATE = r"""
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
"""

DEFAULT_CLAUDE_PROMPT_TEMPLATE = r"""
Tu es un éditeur de style. Réécris et condense le texte fourni en assurant cohérence,
clarté et fluidité, en gardant les citations APA. Harmonise le ton et la terminologie.
"""

@dataclass
class AppConfig:
    drafter_model: str = DEFAULT_DRAFTER_MODEL
    refiner_model: str = DEFAULT_REFINER_MODEL
    api_retry_delays: List[int] = field(default_factory=lambda: API_RETRY_DELAYS.copy())
    styles: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_STYLES))
    default_paths: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PATHS))

    min_relevance_score: int = MIN_RELEVANCE_SCORE
    max_citations_per_section: int = MAX_CITATIONS_PER_SECTION
    include_secondary_matches: bool = INCLUDE_SECONDARY_MATCHES
    confidence_threshold: int = CONFIDENCE_THRESHOLD

    gpt_prompt_template: str = DEFAULT_GPT_PROMPT_TEMPLATE
    claude_prompt_template: str = DEFAULT_CLAUDE_PROMPT_TEMPLATE
    
    # Répertoire d'export
    export_dir: str = "output"
    
    # Champs de style pour l'interface
    font_family: str = "Calibri"
    font_size_body: int = 11
    font_size_h1: int = 18
    font_size_h2: int = 14
    margin_top: float = 2.5
    margin_bottom: float = 2.5
    margin_left: float = 2.5
    margin_right: float = 2.5
    
    # Modèles préférés
    preferred_models: Dict[str, str] = field(default_factory=lambda: {
        "draft": "gpt-5-mini",
        "final": "gpt-5"
    })
    
    # Fichiers par défaut
    default_files: Dict[str, str] = field(default_factory=lambda: {
        "plan": "plan_ouvrage.docx",
        "corpus": "corpus_ANALYZED.xlsx",
        "keywords": "keywords_mapping.json"
    })
    
    # Paramètres de génération pour le brouillon (IA 1)
    draft_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.9,
        "top_p": 0.95,
        "max_output_tokens": 8192,   # Augmenté par rapport à l'ancienne valeur
        "reasoning_effort": "medium", # NOUVEAU pour GPT-5
        "verbosity": "medium"         # NOUVEAU pour GPT-5
    })
    
    # Paramètres de génération pour la version finale (IA 2)
    final_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 16000,  # Augmenté par rapport à l'ancienne valeur
        "reasoning_effort": "high",  # NOUVEAU pour GPT-5
        "verbosity": "medium"         # NOUVEAU pour GPT-5
    })

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_user_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_prompts_yaml(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    res = {}
    if "drafter" in data: res["gpt_prompt_template"] = data["drafter"]
    if "refiner" in data: res["claude_prompt_template"] = data["refiner"]
    return res

def get_config(user_yaml_path: str = "config/user.yaml",
               prompts_yaml_path: str = "config/prompts.yaml") -> AppConfig:
    base = AppConfig().__dict__
    overrides = load_user_yaml(user_yaml_path)
    prompt_overrides = load_prompts_yaml(prompts_yaml_path)
    merged = _deep_update(base.copy(), overrides)
    merged = _deep_update(merged, prompt_overrides)
    return AppConfig(**merged)
