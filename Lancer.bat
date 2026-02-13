@echo off
chcp 65001 >nul 2>&1
title Generateur d'Ouvrage - Lancement

echo ============================================
echo   Generateur d'Ouvrage Assiste par IA
echo ============================================
echo.

REM Se placer dans le repertoire du script
cd /d "%~dp0"

REM Verifier que Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH.
    echo.
    echo Installez Python depuis https://www.python.org/downloads/
    echo Cochez bien "Add Python to PATH" lors de l'installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte.

REM Installer les dependances si necessaire
if exist "requirements.txt" (
    echo.
    echo Verification des dependances...
    pip install -r requirements.txt -q >nul 2>&1
    echo [OK] Dependances installees.
)

echo.
echo Lancement de l'application sur http://localhost:8504
echo Pour arreter : fermez cette fenetre ou appuyez sur Ctrl+C.
echo.

REM Ouvrir le navigateur apres un court delai
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8504"

REM Lancer Streamlit
set PYTHONPATH=%~dp0src;%PYTHONPATH%
streamlit run src/app.py --server.port 8504

pause
