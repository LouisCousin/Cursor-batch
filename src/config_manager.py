
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
                        value = value.strip()
                        # Retirer les guillemets entourant la valeur
                        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                            value = value[1:-1]
                        os.environ[key.strip()] = value
        except Exception as e:
            print(f"Warning: Impossible de charger le fichier .env: {e}")

# Charger les variables d'environnement au démarrage
load_env_vars()

# Catalogues modèles
AVAILABLE_DRAFTER_MODELS = ["GPT-4.1", "GPT-4.1 mini", "GPT-4.1 nano"]
AVAILABLE_REFINER_MODELS = ["Claude 4 Sonnet"]

# Fournisseurs détaillés
AVAILABLE_OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
AVAILABLE_ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3.5-sonnet-20240620"]

MODEL_ALIASES = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "Claude 4 Sonnet": "claude-sonnet-4-20250514",
}

# Limites des modèles : source unique de vérité pour les capacités
MODEL_LIMITS = {
    # Modèle: {"context": total_tokens, "max_output": max_tokens_sortie}
    "gpt-5": {"context": 400000, "max_output": 128000},
    "gpt-5-mini": {"context": 400000, "max_output": 128000},
    "gpt-5-nano": {"context": 400000, "max_output": 128000},
    "gpt-4.1": {"context": 1047576, "max_output": 32768},
    "gpt-4.1-mini": {"context": 1047576, "max_output": 32768},
    "gpt-4.1-nano": {"context": 1047576, "max_output": 32768},
    "claude-sonnet-4-20250514": {"context": 200000, "max_output": 64000},
    "claude-3.5-sonnet-20240620": {"context": 200000, "max_output": 8192},
}

def get_model_config(model: str) -> Dict[str, Any]:
    """
    Retourne la configuration d'un modèle depuis MODEL_LIMITS.
    
    Args:
        model: Nom du modèle
        
    Returns:
        Configuration du modèle avec context et max_output
    """
    return MODEL_LIMITS.get(model, {"context": 4096, "max_output": 1024})



# Defaults
DEFAULT_DRAFTER_MODEL = "GPT-4.1"
DEFAULT_REFINER_MODEL = "Claude 4 Sonnet"
API_RETRY_DELAYS = [60, 300]

DEFAULT_STYLES = {
    "font_family": "Calibri",
    "font_size_body": 11,
    "font_size_h1": 18,
    "font_size_h2": 14,
    "font_size_h3": 12,
    "margin_top": 2.5,
    "margin_bottom": 2.5,
    "margin_left": 2.5,
    "margin_right": 2.5,
    # Interlignes et espacement paragraphes
    "line_spacing": 1.15,
    "space_after_paragraph": 6,   # points
    "space_before_h1": 24,        # points
    "space_after_h1": 12,         # points
    "space_before_h2": 18,        # points
    "space_after_h2": 8,          # points
    "space_before_h3": 12,        # points
    "space_after_h3": 6,          # points
    # Couleurs de titres (RGB hex sans #)
    "heading_color_h1": "1F3864",  # bleu foncé
    "heading_color_h2": "2E5090",  # bleu moyen
    "heading_color_h3": "404040",  # gris foncé
    # Titres en gras
    "heading_bold": True,
    # Format de page
    "page_width": 21.0,   # cm – A4
    "page_height": 29.7,  # cm – A4
    # Première ligne en retrait (cm, 0 = désactivé)
    "first_line_indent": 0,
}

# Chemins par défaut (peuvent être surchargés dans config/user.yaml)
DEFAULT_PATHS = {
    "plan_docx": "",
    "excel_corpus": "",
    "keywords_json": "",
    "env_dir": "",
}

# Corpus filtering (scores on 0-10 scale, confidence on 0-100 scale)
MIN_RELEVANCE_SCORE = 7
MAX_CITATIONS_PER_SECTION = 30
INCLUDE_SECONDARY_MATCHES = True
CONFIDENCE_THRESHOLD = 60

# Normalized corpus filtering (0-1 scale, used by get_relevant_content and UI)
MIN_RELEVANCE_SCORE_NORMALIZED = 0.7
CONFIDENCE_THRESHOLD_NORMALIZED = 0.6

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
    font_size_h3: int = 12
    margin_top: float = 2.5
    margin_bottom: float = 2.5
    margin_left: float = 2.5
    margin_right: float = 2.5
    line_spacing: float = 1.15
    space_after_paragraph: int = 6
    heading_bold: bool = True
    heading_color_h1: str = "1F3864"
    heading_color_h2: str = "2E5090"
    heading_color_h3: str = "404040"
    page_width: float = 21.0
    page_height: float = 29.7
    first_line_indent: float = 0
    
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

class ConfigManager:
    """
    Gestionnaire de configuration centralisé avec support des chemins absolus.
    """
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            config_dir: Répertoire contenant les fichiers de configuration
        """
        # Déterminer la racine du projet de manière dynamique
        # __file__ est le chemin de ce script (config_manager.py dans src/)
        script_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(script_path)
        self.project_root = os.path.dirname(src_dir)
        
        # Chemins de configuration
        self.config_dir = config_dir
        self.user_yaml_path = os.path.join(self.project_root, config_dir, "user.yaml")
        self.prompts_yaml_path = os.path.join(self.project_root, config_dir, "prompts.yaml")
        
        # Charger la configuration
        self.user_config = load_user_yaml(self.user_yaml_path)
        self.app_config = self._load_app_config()
    
    def _load_app_config(self) -> AppConfig:
        """Charge la configuration complète de l'application."""
        base = AppConfig().__dict__
        prompt_overrides = load_prompts_yaml(self.prompts_yaml_path)
        merged = _deep_update(base.copy(), self.user_config)
        merged = _deep_update(merged, prompt_overrides)
        # Filtrer les clés inconnues pour éviter TypeError sur AppConfig
        valid_keys = set(AppConfig().__dict__.keys())
        filtered = {k: v for k, v in merged.items() if k in valid_keys}
        return AppConfig(**filtered)
    
    def get_default_paths(self) -> Dict[str, str]:
        """
        Retourne les chemins par défaut des fichiers en tant que chemins absolus.
        
        Returns:
            Dictionnaire avec les chemins absolus des fichiers par défaut
        """
        default_files = self.user_config.get('default_files', {})
        
        paths = {}
        for key, relative_path in default_files.items():
            if relative_path:
                # Si le chemin est déjà absolu, le garder tel quel
                if os.path.isabs(relative_path):
                    paths[key] = relative_path
                else:
                    # Sinon, le construire depuis la racine du projet
                    paths[key] = os.path.join(self.project_root, relative_path)
            else:
                paths[key] = ""
        
        return paths
    
    def get_project_root(self) -> str:
        """Retourne le chemin racine du projet."""
        return self.project_root
    
    def get_config(self) -> AppConfig:
        """Retourne la configuration de l'application."""
        return self.app_config


def get_config(user_yaml_path: str = "config/user.yaml",
               prompts_yaml_path: str = "config/prompts.yaml") -> AppConfig:
    """
    Fonction de compatibilité pour charger la configuration.
    
    Note: Il est recommandé d'utiliser ConfigManager pour les nouvelles implémentations.
    """
    base = AppConfig().__dict__
    overrides = load_user_yaml(user_yaml_path)
    prompt_overrides = load_prompts_yaml(prompts_yaml_path)
    merged = _deep_update(base.copy(), overrides)
    merged = _deep_update(merged, prompt_overrides)
    # Filtrer les clés inconnues pour éviter TypeError sur AppConfig
    valid_keys = set(AppConfig().__dict__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}
    return AppConfig(**filtered)
