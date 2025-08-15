#!/usr/bin/env python3
"""
Application Streamlit pour la génération d'ouvrages assistée par IA.
Version refondue avec interface simplifiée en 5 pages et support des nouveaux modèles.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import streamlit as st
import pandas as pd

from config_manager import (
    __version__, get_config,
    AVAILABLE_OPENAI_MODELS, AVAILABLE_ANTHROPIC_MODELS, MODEL_ALIASES, MODEL_LIMITS
)
from core.utils import (
    parse_docx_plan, call_openai, call_anthropic, generate_styled_docx,
    extract_used_references_apa, generate_bibliography, truncate_to_tokens,
    export_markdown, export_docx, calculate_max_input_tokens
)
from core.corpus_manager import CorpusManager
from core.prompt_builder import PromptBuilder
from core.orchestrator import GenerationOrchestrator, GenerationTask, TaskStatus, create_linear_dependency_tasks
from core.process_tracker import ProcessTracker, ProcessStatus
from core.export_utils import PromptExporter

# Fonction utilitaire pour importer le module batch
def import_batch_processor():
    """Importe le BatchProcessor depuis le dossier racine."""
    import sys
    import os
    root_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(root_path)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
    from stubs_batch import BatchProcessor
    return BatchProcessor

st.set_page_config(
    page_title="Générateur d'Ouvrage Assisté par IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"### Générateur d'Ouvrage Assisté par IA — version {__version__}")

# Configuration et état de session
cfg = get_config()
ss = st.session_state

# Initialisation des paramètres de session
ss.setdefault("export_dir", cfg.export_dir)
ss.setdefault("draft_params", cfg.draft_params.copy())
ss.setdefault("final_params", cfg.final_params.copy())

# Initialiser le tracker de processus
if 'process_tracker' not in ss:
    ss.process_tracker = ProcessTracker()

# Initialiser l'exporteur de prompts
if 'prompt_exporter' not in ss:
    ss.prompt_exporter = PromptExporter()

# Initialiser les prompts dans la session si ce n'est pas déjà fait
if 'prompt_drafter' not in ss:
    ss.prompt_drafter = cfg.gpt_prompt_template
    ss.prompt_refiner = cfg.claude_prompt_template

# Nouvelle structure de navigation en 6 pages
PAGES = [
    "1. Accueil & Fichiers",
    "2. Configuration", 
    "3. Analyse & Préparation",
    "4. Génération",
    "5. Résultats & Export",
    "6. Historique des Générations",
]

# Fonction principale pour les générations avec indicateurs d'activité
def run_generation(mode: str, prompt: str, provider: str, model: str, params: dict, styles: dict, base_name: str):
    """Exécute une génération avec indicateurs visuels et exports automatiques."""
    
    with st.status("Initialisation…", expanded=True) as status:
        prog = st.progress(0)
        
        status.update(label="Préparation du prompt", state="running")
        
        # Étape 1: Calculer la limite d'entrée dynamique
        max_input_len = calculate_max_input_tokens(model, params["max_output_tokens"])
        
        # Étape 2: Tronquer le prompt en utilisant cette limite
        truncated = truncate_to_tokens(prompt, max_input_len, model=model)
        prog.progress(20)
        
        # Étape 3: Appel LLM avec les bons paramètres
        status.update(label=f"Appel {provider}…", state="running")
        try:
            if provider == "OpenAI":
                text = call_openai(
                    model, truncated,
                    api_key=ss.openai_key,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_output_tokens=params["max_output_tokens"],
                    reasoning_effort=params.get("reasoning_effort", "medium"),
                    verbosity=params.get("verbosity", "medium")
                )
            else:  # Anthropic
                text = call_anthropic(
                    model, truncated,
                    api_key=ss.anthropic_key,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_output_tokens=params["max_output_tokens"]
                )
        except Exception as e:
            status.update(label=f"Erreur lors de l'appel API: {str(e)}", state="error")
            st.error(f"Erreur lors de l'appel API: {str(e)}")
            return None, None, None
        
        prog.progress(70)
        
        # Étape 4: Exports automatiques
        status.update(label="Exports en cours…", state="running")
        export_dir = ss.get("export_dir", "output")
        os.makedirs(export_dir, exist_ok=True)
        
        md_path = export_markdown(text, base_name=base_name, mode=mode, export_dir=export_dir)
        docx_path = export_docx(text, base_name=base_name, mode=mode, export_dir=export_dir, styles=styles)
        
        prog.progress(100)
        status.update(label="Terminé ✅", state="complete")
        
        st.toast(f"{mode.capitalize()} exporté :\n• {md_path}\n• {docx_path}")
        
        return text, md_path, docx_path

# Fonction pour obtenir les paramètres actuels
def get_current_params():
    """Retourne les paramètres actuels du corpus avec des valeurs par défaut."""
    return {
        "min_relevance_score": ss.get("min_relevance_score", 0.7),
        "max_citations_per_section": ss.get("max_citations_per_section", 10),
        "include_secondary_matches": ss.get("include_secondary_matches", True),
        "confidence_threshold": ss.get("confidence_threshold", 0.8)
    }

# Sidebar avec navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Navigation", PAGES, index=0, label_visibility="collapsed")
    
    # Affichage des informations de statut
    st.markdown("---")
    st.subheader("Statut du projet")
    
    plan_ok = bool(ss.get('plan_items'))
    excel_ok = bool(ss.get('cm'))
    keys_ok = bool(ss.get('openai_key')) or bool(ss.get('anthropic_key'))
    
    st.success(f"Plan de l'ouvrage : {'✅ Chargé' if plan_ok else '❌ Manquant'}")
    st.success(f"Corpus de données : {'✅ Chargé' if excel_ok else '❌ Manquant'}")
    st.success(f"Clés API : {'✅ Détectées' if keys_ok else '❌ Manquantes'}")
    
    # Métriques de performance
    if ss.get('generation_results'):
        st.markdown("---")
        st.subheader("📊 Métriques de Performance")
        
        total_sections = len(ss.generation_results)
        completed_sections = len([k for k, v in ss.generation_results.items() if "finale" in v])
        
        st.metric("Sections traitées", f"{completed_sections}/{total_sections}")
        
        if completed_sections > 0:
            completion_rate = (completed_sections / total_sections) * 100
            st.metric("Taux de complétion", f"{completion_rate:.1f}%")
            
            # Calculer le temps moyen par section
            if 'generation_start_time' in ss:
                start_time = datetime.fromisoformat(ss.generation_start_time)
                elapsed = datetime.now() - start_time
                avg_time = elapsed / completed_sections
                st.metric("Temps moyen/section", f"{avg_time.total_seconds()/60:.1f} min")
    
    # Statistiques du corpus
    if ss.get('cm') and ss.get('plan_items'):
        st.markdown("---")
        st.subheader("📚 Statistiques du Corpus")
        
        try:
            total_entries = len(ss.cm.df)
            st.metric("Total entrées", total_entries)
            
            # Compter les sections avec couverture
            covered_sections = 0
            current_params = get_current_params()
            for item in ss.plan_items:
                section_title = item.get('title', '')
                filtered_corpus = ss.cm.get_relevant_content(
                    section_title,
                    min_score=current_params["min_relevance_score"],
                    max_citations=current_params["max_citations_per_section"],
                    include_secondary=current_params["include_secondary_matches"],
                    confidence_threshold=current_params["confidence_threshold"]
                )
                if len(filtered_corpus) > 0:
                    covered_sections += 1
            
            coverage_rate = (covered_sections / len(ss.plan_items)) * 100
            st.metric("Sections couvertes", f"{covered_sections}/{len(ss.plan_items)}")
            st.metric("Taux de couverture", f"{coverage_rate:.1f}%")
            
        except Exception as e:
            st.info("Impossible de calculer les statistiques du corpus")

# Page 1: Accueil & Fichiers
if page == "1. Accueil & Fichiers":
    st.header("1. Accueil & Fichiers")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Chargement des fichiers")
        
        # Chargement du plan de l'ouvrage
        plan_file = st.file_uploader(
            "Plan de l'ouvrage (.docx)",
            type=["docx"],
            help="Sélectionnez le fichier Word contenant le plan de votre ouvrage"
        )
        
        if plan_file:
            try:
                ss.plan_items = parse_docx_plan(plan_file)
                st.success(f"✅ Plan chargé : {len(ss.plan_items)} sections détectées")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du plan : {e}")
        
        # Chargement du corpus enrichi
        corpus_file = st.file_uploader(
            "Corpus enrichi (.xlsx ou .csv)",
            type=["xlsx", "csv"],
            help="Sélectionnez le fichier Excel ou CSV contenant votre corpus analysé"
        )
        
        if corpus_file:
            try:
                if corpus_file.name.endswith('.xlsx'):
                    df = pd.read_excel(corpus_file)
                else:
                    df = pd.read_csv(corpus_file)
                
                ss.cm = CorpusManager.from_dataframe(df)
                st.success(f"✅ Corpus chargé : {len(df)} entrées")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du corpus : {e}")
        
        # Chargement du mapping de mots-clés (optionnel)
        keywords_file = st.file_uploader(
            "Mapping de mots-clés (.json) - Optionnel",
            type=["json"],
            help="Fichier JSON contenant le mapping des mots-clés (optionnel)"
        )
        
        if keywords_file:
            try:
                import json
                ss.keywords_mapping = json.load(keywords_file)
                st.success("✅ Mapping de mots-clés chargé")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du mapping : {e}")
    
    with col2:
        st.subheader("⚡ Chargement rapide")
        
        if st.button("Charger depuis les chemins par défaut", type="primary"):
            try:
                # Chargement automatique depuis config/user.yaml
                user_config = get_config()
                default_files = user_config.default_files
                
                # Charger le plan de l'ouvrage
                if default_files.get("plan") and Path(default_files["plan"]).exists():
                    try:
                        ss.plan_items = parse_docx_plan(default_files["plan"])
                        st.success(f"✅ Plan chargé automatiquement : {len(ss.plan_items)} sections")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement du plan : {e}")
                
                # Charger le corpus enrichi
                if default_files.get("corpus") and Path(default_files["corpus"]).exists():
                    try:
                        if default_files["corpus"].endswith('.xlsx'):
                            df = pd.read_excel(default_files["corpus"])
                        else:
                            df = pd.read_csv(default_files["corpus"])
                        
                        ss.cm = CorpusManager.from_dataframe(df)
                        st.success(f"✅ Corpus chargé automatiquement : {len(df)} entrées")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement du corpus : {e}")
                
                # Charger le mapping de mots-clés
                if default_files.get("keywords") and Path(default_files["keywords"]).exists():
                    try:
                        import json
                        with open(default_files["keywords"], 'r', encoding='utf-8') as f:
                            ss.keywords_mapping = json.load(f)
                        st.success("✅ Mapping de mots-clés chargé automatiquement")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement du mapping : {e}")
                
                # Charger les clés API depuis l'environnement
                if os.getenv("OPENAI_API_KEY"):
                    ss.openai_key = os.getenv("OPENAI_API_KEY")
                    st.success("✅ Clé API OpenAI détectée automatiquement")
                
                if os.getenv("ANTHROPIC_API_KEY"):
                    ss.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    st.success("✅ Clé API Anthropic détectée automatiquement")
                
                st.success("🎉 Chargement automatique terminé !")
                
            except Exception as e:
                st.error(f"Erreur lors du chargement automatique : {e}")

# Page 2: Configuration
elif page == "2. Configuration":
    st.header("2. Configuration Générale")
    
    # Paramètres des modèles d'IA et Prompts
    with st.expander("🤖 Paramètres des modèles d'IA et Prompts", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clés API")
            ss.openai_key = st.text_input(
                "Clé API OpenAI",
                type="password",
                value=ss.get("openai_key", ""),
                help="Votre clé API OpenAI"
            )
            
            ss.anthropic_key = st.text_input(
                "Clé API Anthropic",
                type="password",
                value=ss.get("anthropic_key", ""),
                help="Votre clé API Anthropic"
            )
        
        with col2:
            st.subheader("Sélection des modèles")
            
            # Fournisseur et modèle pour le brouillon
            ss.drafter_provider = st.selectbox(
                "Fournisseur (Brouillon)",
                ["OpenAI", "Anthropic"],
                index=0 if ss.get("drafter_provider") != "Anthropic" else 1,
                key="drafter_provider_selector"
            )
            
            drafter_options = AVAILABLE_OPENAI_MODELS if ss.drafter_provider == "OpenAI" else AVAILABLE_ANTHROPIC_MODELS
            try:
                drafter_index = drafter_options.index(ss.get("drafter_model", drafter_options[0]))
            except ValueError:
                drafter_index = 0
            
            ss.drafter_model = st.selectbox(
                "Modèle (Brouillon)",
                options=drafter_options,
                index=drafter_index,
                key="drafter_model_selector"
            )
            
            # Fournisseur et modèle pour la version finale
            ss.final_provider = st.selectbox(
                "Fournisseur (Version Finale)",
                ["OpenAI", "Anthropic"],
                index=0 if ss.get("final_provider") != "Anthropic" else 1,
                key="final_provider_selector"
            )
            
            final_options = AVAILABLE_OPENAI_MODELS if ss.final_provider == "OpenAI" else AVAILABLE_ANTHROPIC_MODELS
            try:
                final_index = final_options.index(ss.get("final_model", final_options[0]))
            except ValueError:
                final_index = 0
            
            ss.final_model = st.selectbox(
                "Modèle (Version Finale)",
                options=final_options,
                index=final_index,
                key="final_model_selector"
            )

    # Édition des Prompts (session uniquement)
    with st.expander("📝 Édition des Prompts", expanded=False):
        st.info("Les modifications apportées ici sont valables pour la session en cours uniquement.")
        
        ss.prompt_drafter = st.text_area(
            "Prompt Brouillon (Drafter)",
            value=ss.prompt_drafter,
            height=300,
            key="drafter_prompt_editor",
            help="Modifiez ici le template de prompt pour la génération des brouillons."
        )

        ss.prompt_refiner = st.text_area(
            "Prompt Raffinage (Refiner)",
            value=ss.prompt_refiner,
            height=200,
            key="refiner_prompt_editor",
            help="Modifiez ici le template de prompt pour le raffinage des textes."
        )
    
    # Paramètres de génération avancés
    with st.expander("⚙️ Paramètres de Génération Avancés"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paramètres Brouillon (IA 1)")
            
            # Déterminer si les contrôles doivent être désactivés
            selected_drafter_model = ss.get("drafter_model", "").lower()
            is_gpt5_reasoning_drafter = "gpt-5" in selected_drafter_model and "chat" not in selected_drafter_model
            
            ss.draft_params["temperature"] = st.slider(
                "Température",
                0.0, 2.0,
                float(ss.draft_params["temperature"]),
                0.05,
                key="draft_temperature",
                disabled=is_gpt5_reasoning_drafter,
                help="Non applicable aux modèles de raisonnement GPT-5." if is_gpt5_reasoning_drafter else "Contrôle la créativité du modèle"
            )
            
            ss.draft_params["top_p"] = st.slider(
                "Top-p",
                0.0, 1.0,
                float(ss.draft_params["top_p"]),
                0.01,
                key="draft_top_p",
                disabled=is_gpt5_reasoning_drafter,
                help="Non applicable aux modèles de raisonnement GPT-5." if is_gpt5_reasoning_drafter else "Contrôle la diversité des tokens sélectionnés"
            )
            
            # Afficher un message d'information si les contrôles sont désactivés
            if is_gpt5_reasoning_drafter:
                st.info("La Température et le Top P sont désactivés pour les modèles de la famille GPT-5.", icon="ℹ️")
            
            # Calcul dynamique de la limite max_output_tokens
            drafter_model = ss.get("drafter_model", "gpt-4.1")
            max_output_limit = MODEL_LIMITS.get(drafter_model, {}).get("max_output", 32768)
            
            ss.draft_params["max_output_tokens"] = st.number_input(
                "Tokens de sortie max",
                min_value=256,
                max_value=max_output_limit,
                value=min(ss.draft_params["max_output_tokens"], max_output_limit),
                step=256,
                key="draft_max_output",
                help=f"Le modèle {drafter_model} supporte jusqu'à {max_output_limit} tokens en sortie."
            )
            
            # Paramètres spécifiques à OpenAI
            if ss.drafter_provider == "OpenAI":
                ss.draft_params["reasoning_effort"] = st.selectbox(
                    "Effort de raisonnement",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(ss.draft_params.get("reasoning_effort", "medium")),
                    key="draft_reasoning"
                )
                
                ss.draft_params["verbosity"] = st.selectbox(
                    "Verbosité",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(ss.draft_params.get("verbosity", "medium")),
                    key="draft_verbosity"
                )
        
        with col2:
            st.subheader("Paramètres Version Finale (IA 2)")
            
            # Déterminer si les contrôles doivent être désactivés
            selected_final_model = ss.get("final_model", "").lower()
            is_gpt5_reasoning_final = "gpt-5" in selected_final_model and "chat" not in selected_final_model
            
            ss.final_params["temperature"] = st.slider(
                "Température",
                0.0, 2.0,
                float(ss.final_params["temperature"]),
                0.05,
                key="final_temperature",
                disabled=is_gpt5_reasoning_final,
                help="Non applicable aux modèles de raisonnement GPT-5." if is_gpt5_reasoning_final else "Contrôle la créativité du modèle"
            )
            
            ss.final_params["top_p"] = st.slider(
                "Top-p",
                0.0, 1.0,
                float(ss.final_params["top_p"]),
                0.01,
                key="final_top_p",
                disabled=is_gpt5_reasoning_final,
                help="Non applicable aux modèles de raisonnement GPT-5." if is_gpt5_reasoning_final else "Contrôle la diversité des tokens sélectionnés"
            )
            
            # Afficher un message d'information si les contrôles sont désactivés
            if is_gpt5_reasoning_final:
                st.info("La Température et le Top P sont désactivés pour les modèles de la famille GPT-5.", icon="ℹ️")
            
            # Calcul dynamique de la limite max_output_tokens
            final_model = ss.get("final_model", "gpt-4.1")
            max_output_limit = MODEL_LIMITS.get(final_model, {}).get("max_output", 32768)
            
            ss.final_params["max_output_tokens"] = st.number_input(
                "Tokens de sortie max",
                min_value=256,
                max_value=max_output_limit,
                value=min(ss.final_params["max_output_tokens"], max_output_limit),
                step=256,
                key="final_max_output",
                help=f"Le modèle {final_model} supporte jusqu'à {max_output_limit} tokens en sortie."
            )
            
            # Paramètres spécifiques à OpenAI
            if ss.final_provider == "OpenAI":
                ss.final_params["reasoning_effort"] = st.selectbox(
                    "Effort de raisonnement",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(ss.final_params.get("reasoning_effort", "high")),
                    key="final_reasoning"
                )
                
                ss.final_params["verbosity"] = st.selectbox(
                    "Verbosité",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(ss.final_params.get("verbosity", "medium")),
                    key="final_verbosity"
                )
        
        # Afficher un message d'information si des contrôles sont désactivés
        selected_drafter_model = ss.get("drafter_model", "").lower()
        selected_final_model = ss.get("final_model", "").lower()
        is_gpt5_reasoning_drafter = "gpt-5" in selected_drafter_model and "chat" not in selected_drafter_model
        is_gpt5_reasoning_final = "gpt-5" in selected_final_model and "chat" not in selected_final_model
        
        if is_gpt5_reasoning_drafter or is_gpt5_reasoning_final:
            st.info(
                "ℹ️ **Modèles GPT-5 détectés :** Les paramètres 'Température' et 'Top P' ne s'appliquent pas aux modèles de la famille GPT-5 de raisonnement. "
                "Le comportement est contrôlé par les paramètres 'Reasoning Effort' et 'Verbosity'. "
                "Utilisez 'gpt-5-chat-latest' si vous avez besoin des paramètres de température."
            )
    
    # Paramètres du corpus
    with st.expander("📚 Paramètres de Filtrage du Corpus"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_relevance_score = st.slider(
                "Score de pertinence minimum",
                0.0, 1.0,
                float(ss.get("min_relevance_score", 0.7)),
                0.05,
                key="min_relevance_widget"
            )
            
            max_citations_per_section = st.number_input(
                "Citations maximum par section",
                min_value=1,
                max_value=50,
                value=int(ss.get("max_citations_per_section", 10)),
                key="max_citations_widget"
            )
        
        with col2:
            include_secondary_matches = st.checkbox(
                "Inclure les correspondances secondaires",
                value=ss.get("include_secondary_matches", True),
                key="include_secondary_widget"
            )
            
            confidence_threshold = st.slider(
                "Seuil de confiance",
                0.0, 1.0,
                float(ss.get("confidence_threshold", 0.8)),
                0.05,
                key="confidence_threshold_widget"
            )
        
        # Stocker les valeurs dans la session state
        ss.min_relevance_score = min_relevance_score
        ss.max_citations_per_section = max_citations_per_section
        ss.include_secondary_matches = include_secondary_matches
        ss.confidence_threshold = confidence_threshold
    
    # Paramètres d'export
    with st.expander("📄 Paramètres d'Export (DOCX & Nommage)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Styles DOCX")
            font_family = st.text_input(
                "Famille de police",
                value=ss.get("font_family", "Calibri"),
                key="font_family_widget"
            )
            
            font_size_body = st.number_input(
                "Taille police corps",
                min_value=8,
                max_value=24,
                value=int(ss.get("font_size_body", 11)),
                key="font_size_body_widget"
            )
            
            font_size_h1 = st.number_input(
                "Taille police H1",
                min_value=12,
                max_value=36,
                value=int(ss.get("font_size_h1", 18)),
                key="font_size_h1_widget"
            )
            
            font_size_h2 = st.number_input(
                "Taille police H2",
                min_value=10,
                max_value=28,
                value=int(ss.get("font_size_h2", 14)),
                key="font_size_h2_widget"
            )
        
        with col2:
            st.subheader("Marges (cm)")
            margin_top = st.number_input(
                "Marge supérieure",
                min_value=0.5,
                max_value=5.0,
                value=float(ss.get("margin_top", 2.5)),
                step=0.1,
                key="margin_top_widget"
            )
            
            margin_bottom = st.number_input(
                "Marge inférieure",
                min_value=0.5,
                max_value=5.0,
                value=float(ss.get("margin_bottom", 2.5)),
                step=0.1,
                key="margin_bottom_widget"
            )
            
            margin_left = st.number_input(
                "Marge gauche",
                min_value=0.5,
                max_value=5.0,
                value=float(ss.get("margin_left", 2.5)),
                step=0.1,
                key="margin_left_widget"
            )
            
            margin_right = st.number_input(
                "Marge droite",
                min_value=0.5,
                max_value=5.0,
                value=float(ss.get("margin_right", 2.5)),
                step=0.1,
                key="margin_right_widget"
            )
        
        export_dir = st.text_input(
            "Dossier d'export",
            value=ss.get("export_dir", "output"),
            key="export_dir_widget"
        )
        
        # Stocker les valeurs dans la session state
        ss.font_family = font_family
        ss.font_size_body = font_size_body
        ss.font_size_h1 = font_size_h1
        ss.font_size_h2 = font_size_h2
        ss.margin_top = margin_top
        ss.margin_bottom = margin_bottom
        ss.margin_left = margin_left
        ss.margin_right = margin_right
        ss.export_dir = export_dir
    
    # Sauvegarde et chargement des configurations
    with st.expander("💾 Sauvegarde et Chargement des Configurations"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sauvegarder la configuration")
            config_name = st.text_input(
                "Nom de la configuration",
                value=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key="save_config_name"
            )
            
            if st.button("💾 Sauvegarder", key="save_config"):
                try:
                    # Préparer la configuration à sauvegarder
                    config_to_save = {
                        'timestamp': datetime.now().isoformat(),
                        'export_dir': ss.get("export_dir", "output"),
                        'draft_params': ss.get("draft_params", {}),
                        'final_params': ss.get("final_params", {}),
                        'min_relevance_score': ss.get("min_relevance_score", 0.7),
                        'max_citations_per_section': ss.get("max_citations_per_section", 10),
                        'include_secondary_matches': ss.get("include_secondary_matches", True),
                        'confidence_threshold': ss.get("confidence_threshold", 0.8),
                        'font_family': ss.get("font_family", "Calibri"),
                        'font_size_body': ss.get("font_size_body", 11),
                        'font_size_h1': ss.get("font_size_h1", 18),
                        'font_size_h2': ss.get("font_size_h2", 14),
                        'margin_top': ss.get("margin_top", 2.5),
                        'margin_bottom': ss.get("margin_bottom", 2.5),
                        'margin_left': ss.get("margin_left", 2.5),
                        'margin_right': ss.get("margin_right", 2.5),
                        'drafter_provider': ss.get("drafter_provider", "OpenAI"),
                        'drafter_model': ss.get("drafter_model", "gpt-4.1"),
                        'final_provider': ss.get("final_provider", "OpenAI"),
                        'final_model': ss.get("final_model", "gpt-4.1")
                    }
                    
                    # Créer le dossier de sauvegarde
                    config_dir = Path("config/saved")
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Sauvegarder en JSON
                    import json
                    config_file = config_dir / f"{config_name}.json"
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"✅ Configuration sauvegardée : {config_file}")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la sauvegarde : {e}")
        
        with col2:
            st.subheader("Charger une configuration")
            
            # Lister les configurations sauvegardées
            config_dir = Path("config/saved")
            if config_dir.exists():
                config_files = list(config_dir.glob("*.json"))
                if config_files:
                    config_options = [f.name.replace('.json', '') for f in config_files]
                    selected_config = st.selectbox(
                        "Configuration à charger",
                        options=config_options,
                        key="load_config_selector"
                    )
                    
                    if st.button("📂 Charger", key="load_config"):
                        try:
                            config_file = config_dir / f"{selected_config}.json"
                            with open(config_file, 'r', encoding='utf-8') as f:
                                saved_config = json.load(f)
                            
                            # Appliquer la configuration
                            for key, value in saved_config.items():
                                if key != 'timestamp':
                                    ss[key] = value
                            
                            st.success(f"✅ Configuration '{selected_config}' chargée avec succès !")
                            st.rerun()  # Recharger la page pour appliquer les changements
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors du chargement : {e}")
                else:
                    st.info("Aucune configuration sauvegardée trouvée.")
            else:
                st.info("Aucun dossier de sauvegarde trouvé.")

# Page 3: Analyse & Préparation
elif page == "3. Analyse & Préparation":
    st.header("3. Analyse & Préparation")
    
    if not ss.get('plan_items') or not ss.get('cm'):
        st.warning("⚠️ Veuillez d'abord charger le plan et le corpus dans la page 'Accueil & Fichiers'")
        st.stop()
    
    # Tableau de couverture globale
    st.subheader("📊 Couverture Globale du Corpus")
    
    coverage_data = []
    for item in ss.plan_items:
        section_code = item.get('code', '')
        section_title = item.get('title', '')
        
        # Calculer la couverture pour cette section
        try:
            filtered_corpus = ss.cm.get_relevant_content(
                section_title,
                min_score=ss.get("min_relevance_score", 0.7),
                max_citations=ss.get("max_citations_per_section", 10),
                include_secondary=ss.get("include_secondary_matches", True),
                confidence_threshold=ss.get("confidence_threshold", 0.8)
            )
            
            citation_count = len(filtered_corpus)
            avg_score = filtered_corpus['Score'].mean() if len(filtered_corpus) > 0 else 0
            
            # Statut couleur basé sur la couverture
            if citation_count >= 5:
                status = "🟢 Excellente"
            elif citation_count >= 3:
                status = "🟡 Bonne"
            elif citation_count >= 1:
                status = "🟠 Faible"
            else:
                status = "🔴 Aucune"
            
            coverage_data.append({
                "Section": f"{section_code} - {section_title}",
                "Citations": citation_count,
                "Score moyen": f"{avg_score:.2f}" if avg_score > 0 else "N/A",
                "Statut": status
            })
        except Exception as e:
            coverage_data.append({
                "Section": f"{section_code} - {section_title}",
                "Citations": "Erreur",
                "Score moyen": "N/A",
                "Statut": "❌ Erreur"
            })
    
    if coverage_data:
        df_coverage = pd.DataFrame(coverage_data)
        st.dataframe(df_coverage, use_container_width=True)
    
    # Mapping des colonnes
    with st.expander("🔧 Mapping des Colonnes du Corpus"):
        st.info("Configurez ici la correspondance entre les colonnes de votre fichier Excel et les champs attendus par l'application.")
        
        # Cette fonctionnalité sera implémentée selon vos besoins spécifiques
        st.write("Fonctionnalité de mapping des colonnes à implémenter")
    
    # Analyse par section
    st.subheader("🔍 Analyse par Section")
    
    if ss.plan_items:
        section_options = [f"{item.get('code', '')} - {item.get('title', '')}" for item in ss.plan_items]
        selected_section = st.selectbox(
            "Sélectionnez une section à analyser",
            options=section_options,
            key="section_analyzer"
        )
        
        if selected_section:
            try:
                # Extraire le titre de la section sélectionnée
                section_title = selected_section.split(" - ", 1)[1] if " - " in selected_section else selected_section
                
                # Prévisualisation du corpus filtré
                filtered_corpus = ss.cm.get_relevant_content(
                    section_title,
                    min_score=ss.get("min_relevance_score", 0.7),
                    max_citations=ss.get("max_citations_per_section", 10),
                    include_secondary=ss.get("include_secondary_matches", True),
                    confidence_threshold=ss.get("confidence_threshold", 0.8)
                )
                
                st.subheader(f"📋 Corpus Filtré pour : {section_title}")
                st.info(f"Affichage des {len(filtered_corpus)} entrées les plus pertinentes")
                
                if len(filtered_corpus) > 0:
                    st.dataframe(filtered_corpus, use_container_width=True)
                    
                    # Bouton d'analyse automatique avec l'IA
                    if st.button(f"🤖 Analyser automatiquement avec l'IA", key=f"analyze_{section_title}"):
                        if not ss.get('openai_key') and not ss.get('anthropic_key'):
                            st.warning("⚠️ Veuillez configurer une clé API dans la page 'Configuration'")
                        else:
                            with st.status("Analyse en cours...", expanded=True) as status:
                                try:
                                    # Construire le prompt d'analyse
                                    prompt_builder = PromptBuilder()
                                    analysis_prompt = prompt_builder.build_analysis_prompt(section_title, filtered_corpus)
                                    
                                    # Choisir le fournisseur et le modèle
                                    provider = ss.get('drafter_provider', 'OpenAI')
                                    model = ss.get('drafter_model', 'gpt-4.1')
                                    api_key = ss.get('openai_key') if provider == 'OpenAI' else ss.get('anthropic_key')
                                    
                                    if provider == 'OpenAI':
                                        analysis_result = call_openai(
                                            model, analysis_prompt, api_key,
                                            temperature=0.3,  # Faible température pour l'analyse
                                            max_output_tokens=2048
                                        )
                                    else:
                                        analysis_result = call_anthropic(
                                            model, analysis_prompt, api_key,
                                            temperature=0.3,
                                            max_output_tokens=2048
                                        )
                                    
                                    status.update(label="Analyse terminée ✅", state="complete")
                                    
                                    # Afficher le résultat
                                    st.subheader("📊 Analyse Automatique de la Section")
                                    st.markdown(analysis_result)
                                    
                                    # Stocker l'analyse dans la session
                                    if 'section_analyses' not in ss:
                                        ss.section_analyses = {}
                                    ss.section_analyses[section_title] = {
                                        'analysis': analysis_result,
                                        'timestamp': datetime.now().isoformat(),
                                        'corpus_size': len(filtered_corpus)
                                    }
                                    
                                except Exception as e:
                                    status.update(label=f"Erreur lors de l'analyse: {str(e)}", state="error")
                                    st.error(f"❌ Erreur lors de l'analyse automatique : {e}")
                    
                    # Afficher les analyses précédentes si disponibles
                    if ss.get('section_analyses') and section_title in ss.section_analyses:
                        with st.expander("📊 Analyses précédentes", expanded=False):
                            analysis_data = ss.section_analyses[section_title]
                            st.markdown(f"**Analysé le :** {analysis_data['timestamp']}")
                            st.markdown(f"**Taille du corpus :** {analysis_data['corpus_size']} entrées")
                            st.markdown("**Résultat de l'analyse :**")
                            st.markdown(analysis_data['analysis'])
                            
                else:
                    st.warning("Aucune entrée trouvée pour cette section avec les critères actuels.")
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de la section : {e}")

# Page 4: Génération
elif page == "4. Génération":
    st.header("4. Génération")
    
    if not ss.get('plan_items') or not ss.get('cm'):
        st.warning("⚠️ Veuillez d'abord charger le plan et le corpus dans la page 'Accueil & Fichiers'")
        st.stop()
    
    if not ss.get('openai_key') and not ss.get('anthropic_key'):
        st.warning("⚠️ Veuillez configurer au moins une clé API dans la page 'Configuration'")
        st.stop()
    
    # Sélecteur de mode et de traitement
    col1, col2 = st.columns(2)
    
    with col1:
        generation_mode = st.radio(
            "Mode de génération",
            ["Manuel (une section)", "Automatique (plusieurs sections)"],
            key="generation_mode"
        )
    
    with col2:
        processing_type = st.radio(
            "Type de traitement",
            ["Synchrone (temps réel)", "Batch (traitement différé)"],
            key="processing_type",
            help="Le traitement synchrone génère immédiatement. Le traitement par lot utilise l'API Batch d'OpenAI (moins cher, plus lent)."
        )
    
    if generation_mode == "Manuel (une section)":
        # Sélection d'une section unique
        section_options = [f"{item.get('code', '')} - {item.get('title', '')}" for item in ss.plan_items]
        selected_section = st.selectbox(
            "Sélectionnez la section à traiter",
            options=section_options,
            key="single_section_selector"
        )
        
        sections_to_process = [selected_section] if selected_section else []
    else:
        # Sélection de plusieurs sections
        section_options = [f"{item.get('code', '')} - {item.get('title', '')}" for item in ss.plan_items]
        
        # Option pour documenter tout l'ouvrage
        select_all = st.checkbox("Documenter tout l'ouvrage (sélectionner tout)")
        
        default_selection = section_options if select_all else (section_options[:3] if len(section_options) >= 3 else section_options)
        
        selected_sections = st.multiselect(
            "Sélectionnez les sections à traiter",
            options=section_options,
            default=default_selection,
            key="multi_section_selector"
        )
        
        sections_to_process = selected_sections
    
    # Bouton de lancement
    if sections_to_process and st.button("🚀 Lancer la Génération", type="primary"):
        if processing_type == "Batch (traitement différé)":
            st.info("🚀 Lancement du processus de génération par lot...")
            try:
                BatchProcessor = import_batch_processor()
                
                # Initialiser le processeur
                batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                
                # Préparer les sections à traiter
                plan_items = []
                for section_full in sections_to_process:
                    if " - " in section_full:
                        section_code, section_title = section_full.split(" - ", 1)
                    else:
                        section_code, section_title = "SECTION", section_full
                    
                    plan_items.append({
                        'code': section_code,
                        'title': section_title
                    })
                
                # Lancer le processus de batch
                process_id = batch_processor.start_new_batch_process(
                    plan_items=plan_items,
                    corpus_manager=ss.cm,
                    prompt_builder=PromptBuilder(draft_template=ss.prompt_drafter),
                    model=ss.drafter_model,  # Utiliser le modèle sélectionné pour le brouillon
                    corpus_params=get_current_params(),  # Utiliser la fonction existante
                    description=f"Génération de {len(plan_items)} sections"
                )
                
                st.success(f"✅ Processus par lot démarré avec succès ! ID du processus : {process_id}")
                st.info("Vous pouvez suivre sa progression dans la page 'Historique des Générations'.")
                
                # Afficher les détails du processus lancé
                with st.expander("📋 Détails du processus", expanded=False):
                    st.markdown(f"**ID du processus :** `{process_id}`")
                    st.markdown(f"**Nombre de sections :** {len(plan_items)}")
                    st.markdown(f"**Modèle utilisé :** {ss.drafter_model}")
                    st.markdown(f"**Type de traitement :** Batch (API OpenAI)")
                    
                    # Lister les sections
                    st.markdown("**Sections à traiter :**")
                    for item in plan_items:
                        st.markdown(f"- {item['code']} - {item['title']}")

            except Exception as e:
                st.error(f"❌ Erreur lors du lancement du processus par lot : {e}")
                st.exception(e)
        
        elif generation_mode == "Manuel (une section)":
            # Mode manuel : conserver l'ancien comportement
            st.subheader("🔄 Progression de la Génération")
            
            # Enregistrer le temps de début
            ss.generation_start_time = datetime.now().isoformat()
            
            # Préparer les styles pour l'export DOCX
            styles = {
                "font_family": ss.get("font_family", "Calibri"),
                "font_size_body": ss.get("font_size_body", 11),
                "font_size_h1": ss.get("font_size_h1", 18),
                "font_size_h2": ss.get("font_size_h2", 14),
                "margin_top": ss.get("margin_top", 2.5),
                "margin_bottom": ss.get("margin_bottom", 2.5),
                "margin_left": ss.get("margin_left", 2.5),
                "margin_right": ss.get("margin_right", 2.5),
            }
            
            # Initialiser les résultats
            ss.generation_results = ss.get("generation_results", {})
            
            section_full = sections_to_process[0]
            # Extraire le code et le titre de la section
            if " - " in section_full:
                section_code, section_title = section_full.split(" - ", 1)
            else:
                section_code, section_title = "SECTION", section_full
            
            # Construire le prompt pour cette section
            try:
                # Récupérer le corpus filtré
                filtered_corpus = ss.cm.get_relevant_content(
                    section_title,
                    min_score=ss.get("min_relevance_score", 0.7),
                    max_citations=ss.get("max_citations_per_section", 10),
                    include_secondary=ss.get("include_secondary_matches", True),
                    confidence_threshold=ss.get("confidence_threshold", 0.8)
                )
                
                if len(filtered_corpus) == 0:
                    st.warning(f"⚠️ Aucune donnée trouvée pour la section '{section_title}'.")
                else:
                    # Construire le prompt
                    prompt_builder = PromptBuilder(
                        draft_template=ss.prompt_drafter,
                        refine_template=ss.prompt_refiner
                    )
                    prompt = prompt_builder.build_draft_prompt(section_title, filtered_corpus)
                    
                    # Lancer la génération
                    text, md_path, docx_path = run_generation(
                        mode="brouillon",
                        prompt=prompt,
                        provider=ss.drafter_provider,
                        model=ss.drafter_model,
                        params=ss.draft_params,
                        styles=styles,
                        base_name=f"{section_code}_{section_title}"
                    )
                    
                    if text and md_path and docx_path:
                        # Stocker les résultats
                        section_key = f"{section_code}_{section_title}"
                        ss.generation_results[section_key] = {
                            "brouillon": text,
                            "md_path": md_path,
                            "docx_path": docx_path,
                            "section_code": section_code,
                            "section_title": section_title,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.success(f"✅ Section '{section_title}' générée avec succès !")
                        
                        # Boutons de téléchargement
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Télécharger .md",
                                data=text.encode("utf-8"),
                                file_name=os.path.basename(md_path),
                                mime="text/markdown"
                            )
                        
                        with col2:
                            with open(docx_path, "rb") as f:
                                st.download_button(
                                    "📥 Télécharger .docx",
                                    data=f.read(),
                                    file_name=os.path.basename(docx_path),
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                    else:
                        st.error(f"❌ Échec de la génération pour la section '{section_title}'")
                        
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement de la section '{section_title}': {e}")
        
        else:
            # Mode automatique : utiliser l'orchestrateur
            st.subheader("🔄 Génération Automatique avec Orchestrateur")
            
            # Option pour le mode d'exécution
            execution_mode = st.selectbox(
                "Mode d'exécution",
                ["Synchrone (sans warnings)", "Parallèle (peut générer des warnings)"],
                index=0,
                help="Le mode synchrone évite les warnings Streamlit mais traite les sections une par une. Le mode parallèle peut traiter plusieurs sections simultanément."
            )
            
            if execution_mode == "Synchrone (sans warnings)":
                st.info("ℹ️ Mode synchrone sélectionné : les sections seront traitées une par une pour éviter les warnings Streamlit.")
            else:
                st.warning("⚠️ Mode parallèle sélectionné : peut générer des warnings 'missing ScriptRunContext' qui peuvent être ignorés.")
            
            # Initialiser l'état de l'orchestrateur dans la session si nécessaire
            if 'orchestrator_state' not in ss:
                ss.orchestrator_state = {
                    'running': False,
                    'tasks': [],
                    'results': {}
                }
            
            # Enregistrer le temps de début
            ss.generation_start_time = datetime.now().isoformat()
            
            # Créer les tâches avec dépendances linéaires
            tasks = create_linear_dependency_tasks(sections_to_process)
            
            # Créer un conteneur pour l'affichage de la progression
            progress_placeholder = st.empty()
            
            def update_progress_display(tasks_list):
                """Met à jour l'affichage de la progression."""
                try:
                    # Préparer les données pour le DataFrame
                    progress_data = []
                    for task in tasks_list:
                        progress_data.append({
                            "Section": f"{task.section_code} - {task.section_title}",
                            "Statut": task.status.value,
                            "Début": task.start_time.strftime("%H:%M:%S") if task.start_time else "-",
                            "Fin": task.end_time.strftime("%H:%M:%S") if task.end_time else "-",
                            "Erreur": task.error_message[:50] + "..." if task.error_message and len(task.error_message) > 50 else task.error_message or "-"
                        })
                    
                    # Afficher le DataFrame dans le placeholder
                    df_progress = pd.DataFrame(progress_data)
                    with progress_placeholder.container():
                        st.subheader("📊 Suivi en Temps Réel")
                        st.dataframe(df_progress, use_container_width=True)
                        
                        # Statistiques rapides
                        completed = len([t for t in tasks_list if t.status == TaskStatus.TERMINE])
                        failed = len([t for t in tasks_list if t.status == TaskStatus.ECHEC])
                        in_progress = len([t for t in tasks_list if t.status == TaskStatus.EN_COURS])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Terminées", completed)
                        with col2:
                            st.metric("En cours", in_progress)
                        with col3:
                            st.metric("Échecs", failed)
                except Exception as e:
                    # En cas d'erreur dans l'affichage, juste passer
                    # Cela évite de planter l'orchestrateur
                    pass
            
            def generation_function(task: GenerationTask, context: str):
                """Fonction de génération pour l'orchestrateur."""
                try:
                    # Récupérer le corpus filtré pour cette section
                    filtered_corpus = ss.cm.get_relevant_content(
                        task.section_title,
                        min_score=ss.get("min_relevance_score", 0.7),
                        max_citations=ss.get("max_citations_per_section", 10),
                        include_secondary=ss.get("include_secondary_matches", True),
                        confidence_threshold=ss.get("confidence_threshold", 0.8)
                    )
                    
                    if len(filtered_corpus) == 0:
                        return None, "Aucune donnée trouvée", False
                    
                    # Construire le prompt avec le contexte
                    prompt_builder = PromptBuilder(
                        draft_template=ss.prompt_drafter,
                        refine_template=ss.prompt_refiner
                    )
                    prompt = prompt_builder.build_draft_prompt(task.section_title, filtered_corpus)
                    
                    # Ajouter le contexte des sections précédentes si disponible
                    if context:
                        prompt = context + "\n\n" + prompt
                    
                    # Préparer les styles pour l'export DOCX
                    styles = {
                        "font_family": ss.get("font_family", "Calibri"),
                        "font_size_body": ss.get("font_size_body", 11),
                        "font_size_h1": ss.get("font_size_h1", 18),
                        "font_size_h2": ss.get("font_size_h2", 14),
                        "margin_top": ss.get("margin_top", 2.5),
                        "margin_bottom": ss.get("margin_bottom", 2.5),
                        "margin_left": ss.get("margin_left", 2.5),
                        "margin_right": ss.get("margin_right", 2.5),
                    }
                    
                    # Lancer la génération
                    text, md_path, docx_path = run_generation(
                        mode="brouillon",
                        prompt=prompt,
                        provider=ss.drafter_provider,
                        model=ss.drafter_model,
                        params=ss.draft_params,
                        styles=styles,
                        base_name=f"{task.section_code}_{task.section_title}"
                    )
                    
                    if text and md_path and docx_path:
                        # Stocker les résultats dans la session
                        if 'generation_results' not in ss:
                            ss.generation_results = {}
                        
                        section_key = f"{task.section_code}_{task.section_title}"
                        ss.generation_results[section_key] = {
                            "brouillon": text,
                            "md_path": md_path,
                            "docx_path": docx_path,
                            "section_code": task.section_code,
                            "section_title": task.section_title,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Créer un résumé simple pour le contexte
                        summary = text[:500] + "..." if len(text) > 500 else text
                        
                        return text, summary, True
                    else:
                        return None, "Échec de la génération", False
                        
                except Exception as e:
                    return None, f"Erreur: {str(e)}", False
            
            # Créer et configurer l'orchestrateur
            orchestrator = GenerationOrchestrator(tasks, update_progress_display)
            orchestrator.set_generation_function(generation_function)
            
            # Marquer comme en cours d'exécution
            ss.orchestrator_state['running'] = True
            ss.orchestrator_state['tasks'] = tasks
            
            try:
                # Lancer l'orchestrateur selon le mode choisi
                if execution_mode == "Synchrone (sans warnings)":
                    results = orchestrator.run()
                else:
                    results = orchestrator.run_parallel()
                
                # Affichage final des résultats
                st.subheader("🎉 Génération Terminée")
                
                # Statistiques finales
                stats = orchestrator.get_statistics()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", stats['total_tasks'])
                with col2:
                    st.metric("Réussies", stats['completed'])
                with col3:
                    st.metric("Échecs", stats['failed'])
                with col4:
                    st.metric("Taux de réussite", f"{stats['completion_rate']:.1f}%")
                
                if stats['total_execution_time']:
                    st.info(f"⏱️ Temps total d'exécution: {stats['total_execution_time']:.1f} secondes")
                
                if stats['completed'] > 0:
                    st.success("✅ Génération automatique terminée avec succès !")
                    st.info("Consultez la page 'Résultats & Export' pour voir tous les résultats et lancer le raffinage.")
                else:
                    st.error("❌ Aucune section n'a été générée avec succès.")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'exécution de l'orchestrateur : {e}")
            finally:
                # Marquer comme terminé
                ss.orchestrator_state['running'] = False

# Page 5: Résultats & Export
elif page == "5. Résultats & Export":
    st.header("5. Résultats & Export")
    
    if not ss.get("generation_results"):
        st.info("📝 Aucun résultat de génération disponible. Lancez d'abord une génération dans la page 'Génération'.")
        st.stop()
    
    # Navigation par onglets
    tab1, tab2, tab3 = st.tabs(["📝 Brouillons (IA 1)", "✨ Versions Finales (IA 2)", "📚 Document Complet"])
    
    with tab1:
        st.subheader("📝 Brouillons Générés (IA 1)")
        
        # Bouton d'export des prompts
        if st.button("📄 Exporter tous les prompts en Word", key="export_prompts"):
            try:
                exported_path = ss.prompt_exporter.export_generation_results_prompts(
                    ss.generation_results,
                    export_dir=ss.get("export_dir", "output")
                )
                
                if exported_path:
                    st.success(f"✅ Prompts exportés vers : {exported_path}")
                    
                    # Bouton de téléchargement
                    with open(exported_path, "rb") as f:
                        st.download_button(
                            "📥 Télécharger le fichier Word",
                            data=f.read(),
                            file_name=os.path.basename(exported_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.error("❌ Erreur lors de l'export des prompts")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'export : {e}")
        
        st.markdown("---")
        
        for section_key, result in ss.generation_results.items():
            with st.container():
                st.markdown(f"### {result['section_title']}")
                st.markdown(f"**Code section :** {result['section_code']}")
                st.markdown(f"**Généré le :** {result['timestamp']}")
                
                # Afficher le contenu du brouillon
                with st.expander("📖 Contenu du brouillon", expanded=False):
                    st.markdown(result['brouillon'])
                
                # Bouton pour lancer le raffinage
                if st.button(f"🚀 Lancer le raffinage (IA 2)", key=f"refine_{section_key}"):
                    st.info(f"🔄 Lancement du raffinage pour la section '{result['section_title']}'...")
                    
                    try:
                        # Construire le prompt de raffinage
                        prompt_builder = PromptBuilder(
                            draft_template=ss.prompt_drafter,
                            refine_template=ss.prompt_refiner
                        )
                        refine_prompt = prompt_builder.build_refine_prompt(result['brouillon'])
                        
                        # Lancer la génération de la version finale
                        text, md_path, docx_path = run_generation(
                            mode="final",
                            prompt=refine_prompt,
                            provider=ss.final_provider,
                            model=ss.final_model,
                            params=ss.final_params,
                            styles={
                                "font_family": ss.get("font_family", "Calibri"),
                                "font_size_body": ss.get("font_size_body", 11),
                                "font_size_h1": ss.get("font_size_h1", 18),
                                "font_size_h2": ss.get("font_size_h2", 14),
                                "margin_top": ss.get("margin_top", 2.5),
                                "margin_bottom": ss.get("margin_bottom", 2.5),
                                "margin_left": ss.get("margin_left", 2.5),
                                "margin_right": ss.get("margin_right", 2.5),
                            },
                            base_name=f"{result['section_code']}_{result['section_title']}"
                        )
                        
                        if text and md_path and docx_path:
                            # Mettre à jour les résultats
                            ss.generation_results[section_key]["finale"] = text
                            ss.generation_results[section_key]["finale_md_path"] = md_path
                            ss.generation_results[section_key]["finale_docx_path"] = docx_path
                            ss.generation_results[section_key]["finale_timestamp"] = datetime.now().isoformat()
                            
                            st.success(f"✅ Version finale générée avec succès !")
                            
                            # Boutons de téléchargement
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Télécharger .md final",
                                    data=text.encode("utf-8"),
                                    file_name=os.path.basename(md_path),
                                    mime="text/markdown"
                                )
                            
                            with col2:
                                with open(docx_path, "rb") as f:
                                    st.download_button(
                                        "📥 Télécharger .docx final",
                                        data=f.read(),
                                        file_name=os.path.basename(docx_path),
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    )
                        else:
                            st.error("❌ Échec de la génération de la version finale")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors du raffinage : {e}")
                
                st.markdown("---")
    
    with tab2:
        st.subheader("✨ Versions Finales (IA 2)")
        
        finales_disponibles = [k for k, v in ss.generation_results.items() if "finale" in v]
        
        if not finales_disponibles:
            st.info("📝 Aucune version finale disponible. Lancez d'abord le raffinage des brouillons.")
        else:
            for section_key in finales_disponibles:
                result = ss.generation_results[section_key]
                with st.container():
                    st.markdown(f"### {result['section_title']}")
                    st.markdown(f"**Code section :** {result['section_code']}")
                    st.markdown(f"**Raffiné le :** {result['finale_timestamp']}")
                    
                    # Afficher le contenu de la version finale
                    with st.expander("📖 Contenu de la version finale", expanded=False):
                        st.markdown(result['finale'])
                    
                    st.markdown("---")
    
    with tab3:
        st.subheader("📚 Document Complet")
        
        finales_disponibles = [k for k, v in ss.generation_results.items() if "finale" in v]
        
        if not finales_disponibles:
            st.info("📝 Aucune version finale disponible pour la compilation du document complet.")
        else:
            if st.button("🔧 Compiler le document complet", type="primary"):
                st.info("🔄 Compilation du document complet en cours...")
                
                try:
                    # Assembler tous les textes des versions finales
                    document_parts = []
                    bibliography_refs = []
                    
                    for section_key in finales_disponibles:
                        result = ss.generation_results[section_key]
                        section_title = result['section_title']
                        section_code = result['section_code']
                        content = result['finale']
                        
                        # Ajouter l'en-tête de section
                        document_parts.append(f"# {section_code} - {section_title}\n\n")
                        document_parts.append(content)
                        document_parts.append("\n\n---\n\n")
                        
                        # Extraire les références utilisées
                        try:
                            refs = extract_used_references_apa(content)
                            bibliography_refs.extend(refs)
                        except:
                            pass
                    
                    # Générer la bibliographie
                    if bibliography_refs:
                        document_parts.append("# Bibliographie\n\n")
                        try:
                            bibliography = generate_bibliography(bibliography_refs)
                            document_parts.append(bibliography)
                        except:
                            document_parts.append("Bibliographie à compléter manuellement.\n")
                    
                    # Assembler le document complet
                    complete_document = "".join(document_parts)
                    
                    # Afficher l'aperçu
                    st.subheader("📖 Aperçu du Document Complet")
                    with st.expander("Voir le contenu complet", expanded=False):
                        st.markdown(complete_document)
                    
                    # Exporter le document complet
                    export_dir = ss.get("export_dir", "output")
                    os.makedirs(export_dir, exist_ok=True)
                    
                    # Export Markdown
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    md_filename = f"{timestamp}_document_complet.md"
                    md_path = os.path.join(export_dir, md_filename)
                    
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(complete_document)
                    
                    # Export DOCX
                    docx_filename = f"{timestamp}_document_complet.docx"
                    docx_path = os.path.join(export_dir, docx_filename)
                    
                    try:
                        generate_styled_docx(
                            complete_document,
                            docx_path,
                            {
                                "font_family": ss.get("font_family", "Calibri"),
                                "font_size_body": ss.get("font_size_body", 11),
                                "font_size_h1": ss.get("font_size_h1", 18),
                                "font_size_h2": ss.get("font_size_h2", 14),
                                "margin_top": ss.get("margin_top", 2.5),
                                "margin_bottom": ss.get("margin_bottom", 2.5),
                                "margin_left": ss.get("margin_left", 2.5),
                                "margin_right": ss.get("margin_right", 2.5),
                            }
                        )
                        
                        st.success("✅ Document complet compilé et exporté avec succès !")
                        
                        # Boutons de téléchargement
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Télécharger .md complet",
                                data=complete_document.encode("utf-8"),
                                file_name=md_filename,
                                mime="text/markdown"
                            )
                        
                        with col2:
                            with open(docx_path, "rb") as f:
                                st.download_button(
                                    "📥 Télécharger .docx complet",
                                    data=f.read(),
                                    file_name=docx_filename,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du DOCX : {e}")
                        st.info("Le fichier Markdown a été créé avec succès.")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la compilation : {e}")

# Page 6: Historique des Générations
elif page == "6. Historique des Générations":
    st.header("6. Historique des Générations")
    
    # Statistiques globales
    try:
        stats = ss.process_tracker.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total processus", stats['total_processes'])
        with col2:
            st.metric("Terminés", stats['completed_processes'])
        with col3:
            st.metric("En échec", stats['failed_processes'])
        with col4:
            st.metric("Taux moyen", f"{stats['average_completion_rate']:.1f}%")
        
        st.markdown("---")
        
    except Exception as e:
        st.warning(f"Impossible de charger les statistiques : {e}")
    
    # Liste des processus
    st.subheader("📋 Historique des Processus")
    
    try:
        processes = ss.process_tracker.get_all_processes(limit=20)
        
        if not processes:
            st.info("Aucun processus de génération dans l'historique.")
        else:
            for process in processes:
                process_summary = ss.process_tracker.get_process_summary(process['process_id'])
                
                if not process_summary:
                    continue
                
                with st.container():
                    # En-tête du processus
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"### {process_summary['type'].title()} - {process_summary['created_at'][:10]}")
                        if process_summary['description']:
                            st.markdown(f"**Description :** {process_summary['description']}")
                    
                    with col2:
                        # Statut avec couleur
                        status = process_summary['status']
                        if status == ProcessStatus.TERMINE.value:
                            st.success(f"✅ {process_summary['status_text']}")
                        elif status == ProcessStatus.EN_ECHEC.value:
                            st.error(f"❌ {process_summary['status_text']}")
                        elif status == ProcessStatus.EN_COURS.value:
                            st.info(f"🔄 {process_summary['status_text']}")
                        else:
                            st.warning(f"⏳ {process_summary['status_text']}")
                    
                    with col3:
                        # Actions spécifiques au type de processus
                        if process_summary.get('type') == 'batch' and process_summary['status'] == ProcessStatus.EN_COURS.value:
                            if st.button("🔄 Actualiser Statut", key=f"refresh_{process['process_id']}"):
                                try:
                                    BatchProcessor = import_batch_processor()
                                    batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                                    updated_processes = batch_processor.monitor_processes()
                                    if updated_processes:
                                        st.success("Statut du processus mis à jour.")
                                        st.rerun()
                                    else:
                                        st.info("Aucune mise à jour nécessaire.")
                                except Exception as e:
                                    st.error(f"Erreur d'actualisation : {e}")
                        
                        if process_summary['can_resume'] and process_summary.get('type') == 'batch':
                            if st.button("🔄 Reprendre", key=f"resume_{process['process_id']}"):
                                try:
                                    BatchProcessor = import_batch_processor()
                                    batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                                    new_batch_id = batch_processor.resume_failed_process(
                                        process['process_id'],
                                        ss.cm,
                                        PromptBuilder(draft_template=ss.prompt_drafter),
                                        ss.drafter_model,
                                        get_current_params()
                                    )
                                    if new_batch_id:
                                        st.success(f"Processus repris avec le batch ID : {new_batch_id}")
                                        st.rerun()
                                    else:
                                        st.info("Aucune section à reprendre.")
                                except Exception as e:
                                    st.error(f"Erreur lors de la reprise : {e}")
                        
                        if st.button("🗑️ Supprimer", key=f"delete_{process['process_id']}"):
                            if ss.process_tracker.delete_process(process['process_id']):
                                st.success("Processus supprimé")
                                st.rerun()
                            else:
                                st.error("Erreur lors de la suppression")
                    
                    # Détails du processus
                    with st.expander("Voir les détails", expanded=False):
                        # Métriques détaillées
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total sections", process_summary['total_sections'])
                        with col2:
                            st.metric("Terminées", process_summary['completed_sections'])
                        with col3:
                            st.metric("Échecs", process_summary['failed_sections'])
                        with col4:
                            st.metric("Progression", f"{process_summary['completion_rate']:.1f}%")
                        
                        # Liste des sections
                        if process.get('sections'):
                            st.subheader("Sections")
                            sections_data = []
                            for section in process['sections']:
                                status_emoji = {
                                    'succes': '✅',
                                    'echec': '❌', 
                                    'en_cours': '🔄',
                                    'en_attente': '⏳'
                                }
                                
                                sections_data.append({
                                    'Code': section['section_code'],
                                    'Titre': section['section_title'],
                                    'Statut': f"{status_emoji.get(section['status'], '❓')} {section['status']}",
                                    'Erreur': section.get('error_message', '')[:50] + '...' if section.get('error_message') and len(section.get('error_message', '')) > 50 else section.get('error_message', '-')
                                })
                            
                            df_sections = pd.DataFrame(sections_data)
                            st.dataframe(df_sections, use_container_width=True)
                        
                        # Affichage de l'historique des lots pour les processus batch
                        if process.get('batch_history') and process_summary.get('type') == 'batch':
                            st.subheader("Historique des Lots (Batches)")
                            for i, batch_info in enumerate(process['batch_history']):
                                with st.container():
                                    st.markdown(f"**Lot #{i+1}**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.text(f"ID: {batch_info.get('batch_id', 'N/A')}")
                                        st.text(f"Type: {batch_info.get('batch_type', 'generation')}")
                                    with col2:
                                        st.text(f"Créé le: {batch_info.get('created_at', 'N/A')[:19]}")
                                        st.text(f"Statut: {batch_info.get('status', 'submitted')}")
                                    with col3:
                                        sections_count = len(batch_info.get('section_codes', []))
                                        st.text(f"Sections: {sections_count}")
                                        
                                        # Boutons pour vérifier, diagnostiquer et traiter le batch
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            if st.button("🔍 Vérifier", key=f"check_batch_{batch_info.get('batch_id')}_{i}"):
                                                try:
                                                    BatchProcessor = import_batch_processor()
                                                    batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                                                    batch_status = batch_processor.check_batch_status(batch_info['batch_id'])
                                                    
                                                    # Affichage du statut détaillé
                                                    st.markdown(f"**Statut :** {batch_status.get('status', 'inconnu')}")
                                                    
                                                    # Afficher l'estimation de progression si disponible
                                                    estimate = batch_processor.get_batch_completion_estimate(batch_info['batch_id'])
                                                    if estimate:
                                                        st.markdown(f"**Progression :** {estimate['progress_percentage']:.1f}%")
                                                        st.progress(estimate['progress_percentage'] / 100)
                                                        st.markdown(f"**Requêtes :** {estimate['completed_requests']}/{estimate['total_requests']}")
                                                        if estimate.get('estimated_remaining_minutes'):
                                                            st.markdown(f"**Temps restant estimé :** {estimate['estimated_remaining_minutes']:.1f} min")
                                                    
                                                    # Afficher le statut complet en mode développeur
                                                    with st.expander("Détails techniques", expanded=False):
                                                        st.json(batch_status)
                                                        
                                                except Exception as e:
                                                    st.error(f"Erreur : {e}")
                                        
                                        with col_b:
                                            if st.button("🩺 Diagnostic", key=f"diagnose_batch_{batch_info.get('batch_id')}_{i}"):
                                                try:
                                                    BatchProcessor = import_batch_processor()
                                                    batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                                                    diagnosis = batch_processor.diagnose_batch_issues(batch_info['batch_id'])
                                                    
                                                    # Affichage du diagnostic
                                                    st.markdown("**Diagnostic du Batch :**")
                                                    
                                                    if diagnosis.get('issues'):
                                                        st.warning(f"**Problèmes détectés :** {len(diagnosis['issues'])}")
                                                        for issue in diagnosis['issues']:
                                                            st.markdown(f"• {issue}")
                                                    else:
                                                        st.success("Aucun problème détecté")
                                                    
                                                    # Afficher le contenu d'erreur si disponible
                                                    if diagnosis.get('error_content'):
                                                        with st.expander("Détails des erreurs", expanded=False):
                                                            st.code(diagnosis['error_content'][:500], language='text')
                                                    
                                                    # Afficher le diagnostic complet
                                                    with st.expander("Rapport complet", expanded=False):
                                                        st.json(diagnosis)
                                                        
                                                except Exception as e:
                                                    st.error(f"Erreur lors du diagnostic : {e}")
                                        
                                        # Bouton pour traiter ce batch spécifique
                                        # Vérifier si le batch a vraiment été traité avec succès
                                        import os
                                        import glob
                                        
                                        # Vérifier si le batch a vraiment généré des fichiers
                                        # Vérifier dans le processus s'il y a des sections avec result_path
                                        has_generated_files = False
                                        if batch_info.get('status') == 'processed':
                                            # Vérifier les sections de ce batch dans le processus
                                            for section_code in batch_info.get('section_codes', []):
                                                # Chercher la section dans les données du processus
                                                for section in process.get('sections', []):
                                                    if section.get('code') == section_code and section.get('result_path'):
                                                        # Vérifier que le fichier existe vraiment
                                                        if os.path.exists(section['result_path']):
                                                            has_generated_files = True
                                                            break
                                                if has_generated_files:
                                                    break
                                        
                                        # Le batch est vraiment traité s'il a le statut ET des fichiers générés
                                        really_processed = (batch_info.get('status') == 'processed' and has_generated_files)
                                        
                                        if not really_processed:
                                            if st.button("🎯 Traiter ce batch", key=f"process_batch_{batch_info.get('batch_id')}_{i}"):
                                                try:
                                                    BatchProcessor = import_batch_processor()
                                                    batch_processor = BatchProcessor(api_key=ss.openai_key, process_tracker=ss.process_tracker)
                                                    
                                                    # Vérifier d'abord le statut du batch
                                                    batch_status = batch_processor.check_batch_status(batch_info['batch_id'])
                                                    
                                                    if batch_status.get('status') == 'completed':
                                                        # Vérifier s'il y a un fichier de sortie
                                                        batch_details = batch_processor.client.batches.retrieve(batch_info['batch_id'])
                                                        
                                                        if batch_details.output_file_id:
                                                            # Traiter ce batch spécifique
                                                            result = batch_processor.process_batch_results(
                                                                batch_info['batch_id'], 
                                                                export_dir=ss.get("export_dir", "data/output")
                                                            )
                                                            
                                                            # Marquer comme traité dans le tracker
                                                            for batch_data in process.get('batch_history', []):
                                                                if batch_data['batch_id'] == batch_info['batch_id']:
                                                                    batch_data['status'] = 'processed'
                                                                    break
                                                            
                                                            st.success(f"✅ Batch traité avec succès !")
                                                            st.success(f"📊 {result['success_count']} sections générées")
                                                            if result['error_count'] > 0:
                                                                st.warning(f"⚠️ {result['error_count']} erreurs")
                                                            st.info(f"📁 Fichiers sauvegardés dans {result['export_dir']}")
                                                            st.rerun()
                                                        else:
                                                            st.error("❌ Ce batch n'a pas de fichier de sortie (échec total)")
                                                            st.info("💡 Utilisez le bouton 'Reprendre' du processus pour relancer avec les paramètres corrigés")
                                                    else:
                                                        st.warning(f"⏳ Ce batch n'est pas encore terminé (statut: {batch_status.get('status')})")
                                                        
                                                except Exception as e:
                                                    st.error(f"Erreur lors du traitement : {e}")
                                        else:
                                            # Batch vraiment traité avec fichiers générés
                                            # Compter les fichiers générés pour ce batch
                                            generated_files = []
                                            for section_code in batch_info.get('section_codes', []):
                                                # Chercher la section dans les données du processus
                                                for section in process.get('sections', []):
                                                    if section.get('code') == section_code and section.get('result_path'):
                                                        if os.path.exists(section['result_path']):
                                                            generated_files.append(section['result_path'])
                                                        break
                                            
                                            st.success(f"✅ Batch déjà traité ({len(generated_files)} fichiers générés)")
                                            
                                            # Afficher les fichiers générés
                                            if generated_files:
                                                with st.expander("📁 Fichiers générés", expanded=False):
                                                    for file_path in generated_files:
                                                        file_name = os.path.basename(file_path)
                                                        if os.path.exists(file_path):
                                                            file_size = os.path.getsize(file_path)
                                                            st.text(f"• {file_name} ({file_size} bytes)")
                                                        else:
                                                            st.text(f"• {file_name} (fichier manquant)")
                                                        
                                        # Si statut 'processed' mais pas de fichiers, proposer de réinitialiser
                                        if batch_info.get('status') == 'processed' and not has_generated_files:
                                            st.warning("⚠️ Batch marqué comme traité mais aucun fichier trouvé")
                                            if st.button("🔄 Réinitialiser statut", key=f"reset_batch_{batch_info.get('batch_id')}_{i}"):
                                                # Réinitialiser le statut pour permettre un nouveau traitement
                                                for batch_data in process.get('batch_history', []):
                                                    if batch_data['batch_id'] == batch_info['batch_id']:
                                                        batch_data['status'] = 'completed'  # Remettre à completed
                                                        break
                                                st.success("Statut réinitialisé - vous pouvez maintenant retraiter ce batch")
                                                st.rerun()
                                    
                                    st.markdown("---")
                        
                        # Informations système
                        st.subheader("Informations Système")
                        st.text(f"ID Processus: {process['process_id']}")
                        st.text(f"Type: {process.get('type', 'batch')}")
                        st.text(f"Créé le: {process.get('created_at', 'N/A')}")
                        st.text(f"Mis à jour le: {process.get('updated_at', 'N/A')}")
                
                st.markdown("---")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'historique : {e}")
        st.exception(e)
    
    # Actions globales
    st.subheader("🛠️ Actions Globales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Nettoyer l'historique (processus terminés > 30 jours)"):
            try:
                deleted_count = ss.process_tracker.cleanup_old_processes(days_old=30)
                if deleted_count > 0:
                    st.success(f"✅ {deleted_count} anciens processus supprimés")
                    st.rerun()
                else:
                    st.info("Aucun processus ancien à supprimer")
            except Exception as e:
                st.error(f"Erreur lors du nettoyage : {e}")
    
    with col2:
        if st.button("🔄 Actualiser"):
            st.rerun()
    
    st.info("💡 **Nouveau :** Utilisez les boutons '🎯 Traiter ce batch' individuels pour chaque batch selon vos besoins.")

# Footer
st.markdown("---")
st.markdown(f"*Application développée avec Streamlit - Version {__version__}*")
