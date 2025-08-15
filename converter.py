#!/usr/bin/env python3
"""
Module de conversion de Markdown vers Docx.
Utilise pypandoc pour convertir le contenu Markdown en fichiers Word.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False
    logging.warning("pypandoc non disponible. Fonctionnalités de conversion Docx désactivées.")


def convert_md_to_docx(markdown_content: str, output_path: str) -> bool:
    """
    Convertit un contenu Markdown en fichier Docx.
    
    Args:
        markdown_content: Le contenu Markdown à convertir
        output_path: Le chemin de sortie pour le fichier .docx
        
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
        
        # Convertir le Markdown en Docx
        pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            outputfile=output_path,
            extra_args=[
                '--reference-doc=',  # Utilise le template par défaut de pandoc
                '--wrap=preserve'     # Préserve les sauts de ligne
            ]
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
