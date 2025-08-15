import sys
from pathlib import Path

import pandas as pd

# Assurer que 'src' est dans le path
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.prompt_builder import PromptBuilder


def test_prompt_builder_uses_session_state_prompts():
    custom_draft = "Ceci est mon prompt de test pour le brouillon : {section_title}"
    custom_refine = "Ceci est mon prompt de test pour le raffinage."

    # Instancier le PromptBuilder avec des prompts personnalisés
    prompt_builder = PromptBuilder(
        draft_template=custom_draft,
        refine_template=custom_refine,
    )

    # Construire un prompt de brouillon avec un DataFrame vide
    df = pd.DataFrame()
    final_prompt = prompt_builder.build_draft_prompt(
        section_title="Test Section",
        corpus_df=df,
    )

    assert "Ceci est mon prompt de test pour le brouillon" in final_prompt
    assert "Test Section" in final_prompt

