#!/usr/bin/env python3
"""
Utility script to make working with the LISTA project easier.
Provides convenient commands for training, evaluating, and visualizing.

Usage:
    python run.py --help                  # Show all available commands
    python run.py train mse_aug           # Train the mse_aug configuration
    python run.py evaluate runs/mse/...   # Evaluate a run
    python run.py setup-check             # Validate setup (same as setup_validation.py)
"""

import argparse
import subprocess
import sys
from pathlib import Path

def train(args):
    """Run training."""
    config = args.config
    
    cmd = [sys.executable, "scriptsv2/train.py", "--config", config]

    if args.num_epochs:
        cmd.extend(["--num_epochs", str(args.num_epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"Iniciando entrenamiento: {' '.join(cmd)}")
    subprocess.run(cmd)

def evaluate(args):
    """Run evaluation."""
    run_dir = args.run_dir
    
    cmd = [sys.executable, "scriptsv2/evaluate.py", "--run_dir", run_dir]
    
    if args.eval_metric:
        cmd.extend(["--eval_metric", args.eval_metric])
    
    print(f"Iniciando evaluación: {' '.join(cmd)}")
    subprocess.run(cmd)

def visualize(args):
    """Run visualization."""
    run_dir = args.run_dir
    
    cmd = [sys.executable, "scriptsv2/visualize.py", "--run_dir", run_dir]
    
    if args.loss_type:
        cmd.extend(["--loss_type", args.loss_type])
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    print(f"Iniciando visualización: {' '.join(cmd)}")
    subprocess.run(cmd)

def setup_check(args):
    """Validate the project setup."""
    print("Ejecutando validación de setup...")
    subprocess.run([sys.executable, "setup_validation.py"])

def compare_runs(args):
    """Compare multiple runs."""
    cmd = [sys.executable, "scriptsv2/compare_runs.py"]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"Comparando runs: {' '.join(cmd)}")
    subprocess.run(cmd)

def list_runs(args):
    """List all available runs."""
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("No hay runs aún.")
        return
    
    print("\nRuns disponibles:")
    print("="*80)
    
    for loss_type in ["mse", "mae", "ssim"]:
        loss_dir = runs_dir / loss_type
        if loss_dir.exists():
            for run_dir in sorted(loss_dir.iterdir()):
                if run_dir.is_dir():
                    checkpoints = (run_dir / "checkpoints").exists()
                    results = (run_dir / "results").exists()
                    status = "✓" if (checkpoints and results) else "⚠"
                    print(f"{status} {run_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Utilidad para el proyecto LISTA Sparse Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run.py train mse_aug                     # Entrenar con MSE + augmentation
  python run.py train mae_aug --num_epochs 1000   # Entrenar con MAE, 1000 epochs
  python run.py evaluate runs/mse/mse_aug_...     # Evaluar un run específico
  python run.py list-runs                         # Listar todos los runs
  python run.py setup-check                       # Validar que todo esté configurado
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    train_parser = subparsers.add_parser("train", help="Entrenar modelo")
    train_parser.add_argument("config", choices=["mse_aug", "mse_no_aug", "mae_aug", "mae_no_aug", "ssim_aug", "ssim_no_aug"],
                             help="Configuración de entrenamiento")
    train_parser.add_argument("--num_epochs", type=int, help="Número de epochs")
    train_parser.add_argument("--batch_size", type=int, help="Tamaño del batch")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--resume", help="Reanudar desde checkpoint")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluar modelo")
    eval_parser.add_argument("run_dir", help="Directorio del run a evaluar")
    eval_parser.add_argument("--eval_metric", choices=["mse", "mae", "ssim", "all"],
                            help="Métrica de evaluación")
    eval_parser.set_defaults(func=evaluate)

    vis_parser = subparsers.add_parser("visualize", help="Visualizar resultados")
    vis_parser.add_argument("run_dir", help="Directorio del run a visualizar")
    vis_parser.add_argument("--loss_type", help="Tipo de loss a visualizar")
    vis_parser.add_argument("--max_samples", type=int, help="Máximo número de muestras")
    vis_parser.set_defaults(func=visualize)

    compare_parser = subparsers.add_parser("compare-runs", help="Comparar runs")
    compare_parser.add_argument("--output", help="Archivo de salida")
    compare_parser.set_defaults(func=compare_runs)

    list_parser = subparsers.add_parser("list-runs", help="Listar runs disponibles")
    list_parser.set_defaults(func=list_runs)

    setup_parser = subparsers.add_parser("setup-check", help="Validar setup")
    setup_parser.set_defaults(func=setup_check)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
