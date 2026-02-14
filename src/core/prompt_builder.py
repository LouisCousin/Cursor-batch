
from typing import List, Dict, Any
import pandas as pd
from config_manager import DEFAULT_GPT_PROMPT_TEMPLATE

class PromptBuilder:
    def __init__(self, draft_template: str = DEFAULT_GPT_PROMPT_TEMPLATE, refine_template: str = "Réécris ce texte en le condensant et en unifiant le style."):
        self.draft_template = draft_template
        self.refine_template = refine_template

    def build_draft_prompt(self, section_title: str, corpus_df: pd.DataFrame, 
                           keywords: List[str] = None, previous_summaries: str = "", 
                           stats: Dict[str, Any] = None) -> str:
        """
        Construit un prompt pour la génération de brouillon.
        
        Args:
            section_title: Titre de la section
            corpus_df: DataFrame du corpus filtré
            keywords: Liste des mots-clés
            previous_summaries: Résumés des sections précédentes
            stats: Statistiques supplémentaires
        
        Returns:
            Prompt formaté pour l'IA
        """
        # Formater le corpus
        corpus_entries = []
        for _, row in corpus_df.iterrows():
            # Détecter la colonne de texte
            text_col = None
            for col in ['Texte', 'Extrait', 'Citation', 'Content', 'Text']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    text_col = col
                    break
            
            if text_col:
                text = str(row[text_col]).strip()
                # Ajouter des métadonnées si disponibles
                metadata = []
                if 'Score' in row and pd.notna(row['Score']):
                    metadata.append(f"Score: {row['Score']:.2f}")
                if 'MatchType' in row and pd.notna(row['MatchType']):
                    metadata.append(f"Type: {row['MatchType']}")
                if 'Confidence' in row and pd.notna(row['Confidence']):
                    metadata.append(f"Confiance: {row['Confidence']}%")
                
                if metadata:
                    corpus_entries.append(f"- {text} ({', '.join(metadata)})")
                else:
                    corpus_entries.append(f"- {text}")
        
        corpus_text = "\n".join(corpus_entries) if corpus_entries else "Aucun corpus disponible pour cette section."
        
        # Formater les mots-clés
        kw = ", ".join(keywords or [])
        
        # Calculer les statistiques
        if stats is None:
            stats = {}
        
        if 'corpus_count' not in stats:
            stats['corpus_count'] = len(corpus_df)
        
        if 'avg_score' not in stats and len(corpus_df) > 0:
            if 'Score' in corpus_df.columns:
                stats['avg_score'] = f"{corpus_df['Score'].mean():.2f}"
            else:
                stats['avg_score'] = "N/A"
        
        # Construire le prompt
        prompt = self.draft_template
        prompt = prompt.replace("{section_title}", section_title)
        prompt = prompt.replace("{section_plan}", section_title)  # Utiliser le titre comme plan par défaut
        prompt = prompt.replace("{corpus}", corpus_text)
        prompt = prompt.replace("{keywords_found}", kw)
        prompt = prompt.replace("{corpus_count}", str(stats.get("corpus_count", 0)))
        prompt = prompt.replace("{avg_score}", str(stats.get("avg_score", "N/A")))
        prompt = prompt.replace("{previous_summaries}", previous_summaries or "")
        
        return prompt

    def build_refine_prompt(self, draft_markdown: str, style_guidelines: str = "",
                           section_title: str = "", section_code: str = "") -> str:
        """
        Construit un prompt pour le raffinage du texte.

        Args:
            draft_markdown: Texte du brouillon à raffiner
            style_guidelines: Consignes de style spécifiques
            section_title: Titre de la section (pour le contexte)
            section_code: Code de la section (pour le contexte)

        Returns:
            Prompt formaté pour le raffinage
        """
        base_prompt = self.refine_template

        # Ajouter le contexte de la section si disponible
        if section_title or section_code:
            context = f"\n\n## Contexte de la section\n"
            if section_code:
                context += f"- Code : {section_code}\n"
            if section_title:
                context += f"- Titre : {section_title}\n"
            base_prompt += context

        if style_guidelines:
            base_prompt += f"\n\n## Consignes de style spécifiques\n{style_guidelines}"

        return f"{base_prompt}\n\n---\n\n{draft_markdown}"
    
    def build_analysis_prompt(self, section_title: str, corpus_df: pd.DataFrame) -> str:
        """
        Construit un prompt pour l'analyse d'une section.
        
        Args:
            section_title: Titre de la section
            corpus_df: DataFrame du corpus
        
        Returns:
            Prompt formaté pour l'analyse
        """
        prompt = f"""
# Analyse de la section : {section_title}

## Objectif
Analyser la couverture et la pertinence du corpus pour cette section.

## Corpus disponible
- Nombre d'entrées : {len(corpus_df)}
- Colonnes disponibles : {', '.join(corpus_df.columns.tolist())}

## Consignes d'analyse
1. Évaluer la qualité et la pertinence du corpus
2. Identifier les forces et faiblesses
3. Suggérer des améliorations si nécessaire
4. Fournir un score de couverture global (0-10)

## Données du corpus
{corpus_df.head(10).to_string() if len(corpus_df) > 0 else "Aucune donnée disponible"}

## Analyse demandée
"""
        return prompt
