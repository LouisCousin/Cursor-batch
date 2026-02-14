#!/usr/bin/env python3
"""
Module de conversion de Markdown vers Docx.
Utilise pypandoc pour convertir le contenu Markdown en fichiers Word.
Supporte un document de référence (--reference-doc) pour appliquer
des styles professionnels cohérents.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False
    logging.warning("pypandoc non disponible. Fonctionnalités de conversion Docx désactivées.")


# Chemin par défaut du template de référence
_DEFAULT_REFERENCE_DOC = Path(__file__).parent / "config" / "reference.docx"


def _build_reference_doc(dest: Path, styles: dict = None) -> Path:
    """Crée un document de référence Word avec les styles configurés.

    Délègue la configuration des styles à utils.py (source unique de vérité)
    pour éviter la duplication de code.

    Args:
        dest: Chemin de destination du fichier de référence.
        styles: Dictionnaire de styles (clés identiques à DEFAULT_STYLES).

    Returns:
        Chemin du fichier de référence créé.
    """
    if not DOCX_AVAILABLE:
        logging.warning("python-docx non disponible, impossible de créer le reference.docx")
        return dest

    from config_manager import DEFAULT_STYLES
    from src.core.utils import (
        _apply_doc_styles, _setup_normal_style, _setup_heading_style,
        _setup_list_styles,
    )

    s = lambda key: (styles or {}).get(key, DEFAULT_STYLES.get(key))

    font = s("font_family") or "Calibri"
    body_size = int(s("font_size_body") or 11)
    h1_size = int(s("font_size_h1") or 18)
    h2_size = int(s("font_size_h2") or 14)
    h3_size = int(s("font_size_h3") or 12)
    line_spacing = float(s("line_spacing") or 1.15)
    bold = bool(s("heading_bold") if s("heading_bold") is not None else True)
    space_after = int(s("space_after_paragraph") or 6)
    first_indent = float(s("first_line_indent") or 0)

    doc = Document()

    # Marges et format de page (source unique : utils._apply_doc_styles)
    _apply_doc_styles(doc, styles or {})

    # Style Normal
    _setup_normal_style(doc, font, body_size, line_spacing, space_after, first_indent)

    # Headings 1-3
    _setup_heading_style(doc, "Heading 1", font, h1_size,
                         s("heading_color_h1") or "1F3864", bold,
                         int(s("space_before_h1") or 24), int(s("space_after_h1") or 12))
    _setup_heading_style(doc, "Heading 2", font, h2_size,
                         s("heading_color_h2") or "2E5090", bold,
                         int(s("space_before_h2") or 18), int(s("space_after_h2") or 8))
    _setup_heading_style(doc, "Heading 3", font, h3_size,
                         s("heading_color_h3") or "404040", bold,
                         int(s("space_before_h3") or 12), int(s("space_after_h3") or 6))

    # Listes
    _setup_list_styles(doc, font, body_size, line_spacing)

    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dest))
    return dest


def convert_md_to_docx(markdown_content: str, output_path: str,
                       reference_doc: str = None, styles: dict = None) -> bool:
    """
    Convertit un contenu Markdown en fichier Docx avec styles professionnels.

    Si un document de référence est fourni (ou si des styles sont passés),
    les styles Word du document de référence sont appliqués automatiquement
    par Pandoc, ce qui donne un rendu nettement supérieur (polices, interlignes,
    couleurs de titres, espacements).

    Args:
        markdown_content: Le contenu Markdown à convertir
        output_path: Le chemin de sortie pour le fichier .docx
        reference_doc: Chemin vers un document Word de référence (optionnel).
                       Si None et styles fourni, un template est généré automatiquement.
        styles: Dictionnaire de styles pour générer le reference doc à la volée.

    Returns:
        True si la conversion a réussi, False sinon
    """
    if not PYPANDOC_AVAILABLE:
        logging.error("pypandoc n'est pas installé. Impossible de convertir en Docx.")
        return False

    try:
        # Créer le répertoire de sortie si nécessaire
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        extra_args = ['--wrap=preserve']

        # Gestion du document de référence
        ref_path = None
        if reference_doc and Path(reference_doc).exists():
            ref_path = reference_doc
        elif styles and DOCX_AVAILABLE:
            # Générer un template à la volée
            auto_ref = Path(output_path).parent / ".reference_template.docx"
            _build_reference_doc(auto_ref, styles)
            ref_path = str(auto_ref)
        elif _DEFAULT_REFERENCE_DOC.exists():
            ref_path = str(_DEFAULT_REFERENCE_DOC)

        if ref_path:
            extra_args.append(f'--reference-doc={ref_path}')

        # Convertir le Markdown en Docx
        pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            outputfile=output_path,
            extra_args=extra_args
        )

        logging.info(f"Conversion Docx réussie : {output_path}")
        return True

    except Exception as e:
        logging.error(f"Erreur lors de la conversion Docx vers {output_path}: {e}")
        return False


def ensure_pypandoc_installed() -> bool:
    """
    Vérifie si pypandoc est installé et fonctionne correctement.

    Returns:
        True si pypandoc est disponible et fonctionne, False sinon
    """
    if not PYPANDOC_AVAILABLE:
        return False

    try:
        # Test simple de conversion
        test_content = "# Test\n\nCeci est un test."
        result = pypandoc.convert_text(test_content, 'html', format='md')
        return len(result) > 0
    except Exception as e:
        logging.error(f"pypandoc n'est pas correctement configuré : {e}")
        return False


def get_conversion_info() -> dict:
    """
    Retourne des informations sur la disponibilité de la conversion.

    Returns:
        Dictionnaire avec les informations de disponibilité
    """
    info = {
        "pypandoc_available": PYPANDOC_AVAILABLE,
        "conversion_enabled": PYPANDOC_AVAILABLE and ensure_pypandoc_installed()
    }

    if PYPANDOC_AVAILABLE:
        try:
            info["pypandoc_version"] = pypandoc.get_pandoc_version()
        except Exception:
            info["pypandoc_version"] = "inconnue"

    return info
