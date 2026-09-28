#!/usr/bin/env python3
"""
Setup validation script for the LISTA project.
Checks that all paths, data, and modules are correctly configured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scriptsv2"))

def check_structure():
    """Check the project's folder structure."""
    print("\n" + "="*60)
    print("VALIDACIÓN DE ESTRUCTURA DE CARPETAS")
    print("="*60)
    
    root = Path(__file__).resolve().parent
    checks = {
        "data/cells": root / "data" / "cells",
        "data/cells/mitotic_figure": root / "data" / "cells" / "mitotic_figure",
        "data/cells/not_mitotic_figure": root / "data" / "cells" / "not_mitotic_figure",
        "data_reference/masks_bbox_cells": root / "data_reference" / "masks_bbox_cells",
        "runs": root / "runs",
        "scriptsv2": root / "scriptsv2",
    }
    
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name:40} {'OK' if exists else 'FALTA'}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_data():
    """Check the amount of available data."""
    print("\n" + "="*60)
    print("VALIDACIÓN DE DATOS")
    print("="*60)
    
    root = Path(__file__).resolve().parent
    
    classes = {
        "mitotic_figure": root / "data" / "cells" / "mitotic_figure",
        "not_mitotic_figure": root / "data" / "cells" / "not_mitotic_figure",
    }
    
    total_images = 0
    for class_name, path in classes.items():
        if path.exists():
            images = list(path.glob("*.png"))
            count = len(images)
            total_images += count
            print(f"✓ {class_name:25} {count:6} imágenes")
        else:
            print(f"✗ {class_name:25} FALTA")
    
    print(f"\nTotal de imágenes: {total_images}")
    return total_images > 0

def check_modules():
    """Check that the main modules load correctly."""
    print("\n" + "="*60)
    print("VALIDACIÓN DE MÓDULOS")
    print("="*60)
    
    modules = [
        "configuration",
        "dataset",
        "model",
        "loss",
        "train",
        "evaluate",
        "visualize",
    ]
    
    all_ok = True
    for mod_name in modules:
        try:
            __import__(mod_name)
            print(f"✓ {mod_name:30} OK")
        except ImportError as e:
            print(f"✗ {mod_name:30} ERROR: {e}")
            all_ok = False
        except Exception as e:
            print(f"⚠ {mod_name:30} WARNING: {e}")
    
    return all_ok

def check_dataset_loading():
    """Check that the dataset can be loaded."""
    print("\n" + "="*60)
    print("VALIDACIÓN DE DATASET")
    print("="*60)
    
    try:
        import configuration as cfg
        from dataset import CellsDataset
        
        print(f"Intentando cargar dataset desde: {cfg.CELLS_DIR}")
        dataset = CellsDataset(cfg.CELLS_DIR, augment=False)
        print(f"✓ Dataset cargado exitosamente")
        print(f"  Total de muestras: {len(dataset)}")
        print(f"  Clases: {dataset.labels.unique().tolist()}")
        
        sample = dataset[0]
        print(f"  Dimensiones de muestra: {sample[0].shape}")
        print(f"  Rango de valores: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Error al cargar dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model():
    """Check that the model can be instantiated."""
    print("\n" + "="*60)
    print("VALIDACIÓN DE MODELO")
    print("="*60)
    
    try:
        import torch
        import configuration as cfg
        from model import LinearLISTAEncoder, LinearLISTADecoder
        
        encoder = LinearLISTAEncoder(cfg.IN_DIM, cfg.CODE_DIM, cfg.NUM_ITERS)
        decoder = LinearLISTADecoder(encoder)
        
        print(f"✓ Modelos instanciados correctamente")
        print(f"  Encoder: IN_DIM={cfg.IN_DIM}, CODE_DIM={cfg.CODE_DIM}, NUM_ITERS={cfg.NUM_ITERS}")
        print(f"  Parámetros encoder: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  Parámetros decoder: {sum(p.numel() for p in decoder.parameters()):,}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, cfg.IN_DIM).to(device)
        encoder.to(device)
        decoder.to(device)
        
        with torch.no_grad():
            code = encoder(x)
            recon = decoder(code)
        
        print(f"✓ Forward pass exitoso")
        print(f"  Entrada: {x.shape}")
        print(f"  Código: {code.shape}")
        print(f"  Reconstrucción: {recon.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error al instanciar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "█"*60)
    print("█ VALIDACIÓN DE SETUP - LISTA Sparse Autoencoder")
    print("█"*60)
    
    results = {}
    results["structure"] = check_structure()
    results["data"] = check_data()
    results["modules"] = check_modules()
    results["dataset"] = check_dataset_loading()
    results["model"] = check_model()
    
    print("\n" + "="*60)
    print("RESUMEN DE VALIDACIÓN")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✓ OK" if result else "✗ FALLA"
        print(f"{status:5} - {check_name}")
    
    all_ok = all(results.values())
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ ¡SETUP CORRECTO! Listo para entrenar.")
        print("\nPuedes empezar con:")
        print("  cd /home/enrique.pardo.garcia/MIDOGpp")
        print("  source .venv/bin/activate")
        print("  python scriptsv2/train.py --config mse_aug")
    else:
        print("✗ Hay problemas en el setup. Revisa los errores arriba.")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
