#!/usr/bin/env python3
"""
Script de lancement pour l'application Streamlit.
Ce script configure l'environnement Python et lance l'application.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Configure l'environnement Python pour le lancement de l'application."""
    # Obtenir le répertoire racine du projet
    project_root = Path(__file__).parent.absolute()
    
    # Ajouter le répertoire src au path Python
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Ajouter le répertoire racine au path Python
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le répertoire de travail vers le répertoire racine
    os.chdir(project_root)
    
    print(f"✅ Environnement configuré:")
    print(f"   Répertoire racine: {project_root}")
    print(f"   Répertoire src ajouté au path: {src_path}")
    print(f"   Répertoire de travail: {os.getcwd()}")

def main():
    """Fonction principale de lancement."""
    print("🚀 Lancement de l'application Streamlit...")
    
    # Configurer l'environnement
    setup_environment()
    
    # Importer et lancer l'application
    try:
        import streamlit as st
        print("✅ Streamlit importé avec succès")
        
        # Lancer l'application
        print("🌐 Lancement de l'application...")
        os.system("streamlit run src/app.py --server.port 8504")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Assurez-vous que Streamlit est installé: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()