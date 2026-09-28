#!/bin/bash
# Script para iniciar rápidamente el proyecto

cd /home/enrique.pardo.garcia/MIDOGpp
source .venv/bin/activate

# Validar setup
echo "═══════════════════════════════════════════════════════════════"
echo "  LISTA Sparse Autoencoder - Setup Validation"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python setup_validation.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Comandos disponibles para empezar:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Entrenar modelo (MSE con augmentation - RECOMENDADO):"
echo "   python scriptsv2/train.py --config mse_aug"
echo ""
echo "2. O usar el script de utilidad:"
echo "   python run.py train mse_aug"
echo "   python run.py train mae_aug"
echo "   python run.py train ssim_aug"
echo ""
echo "3. Ver lista de runs:"
echo "   python run.py list-runs"
echo ""
echo "4. Evaluar un modelo:"
echo "   python run.py evaluate runs/mse/<run_dir>"
echo ""
echo "5. Ver ayuda:"
echo "   python run.py --help"
echo ""
