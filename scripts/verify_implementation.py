"""
Lightweight verification script that checks file structure and syntax only.
No external dependencies required.
"""

import sys
from pathlib import Path
import ast

# Project root
project_root = Path(__file__).parent.parent


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    path = project_root / file_path
    return path.exists()


def check_python_syntax(file_path: str) -> tuple:
    """Check if a Python file has valid syntax."""
    path = project_root / file_path
    if not path.exists():
        return False, "File not found"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Run verification checks."""
    print("=" * 80)
    print("Phase 0-3 Implementation Verification")
    print("=" * 80 + "\n")
    
    # Core Python files to check
    core_files = [
        # Encoders
        "src/models/encoders/cnn_encoder.py",
        "src/models/encoders/mamba_encoder.py",
        "src/models/encoders/hybrid_encoder.py",
        "src/models/encoders/freq_stream_encoder.py",
        "src/models/encoders/dual_stream_encoder.py",
        "src/models/encoders/__init__.py",
        # Fusion
        "src/models/fusion/gmu.py",
        "src/models/fusion/__init__.py",
        # Heads
        "src/models/heads/mlp_head.py",
        "src/models/heads/wavkan_head.py",
        "src/models/heads/__init__.py",
        # Training modules
        "src/models/moco_module.py",
        "src/models/dual_modal_moco_module.py",
        "src/models/classifier_module.py",
        "src/models/__init__.py",
        # Data
        "src/data/datamodules/pretrain_datamodule.py",
        "src/data/datamodules/finetune_datamodule.py",
        "src/data/datamodules/dual_modal_datamodule.py",
        "src/data/datamodules/__init__.py",
        "src/data/datasets/physionet_dataset.py",
        "src/data/datasets/augmentations.py",
        "src/data/datasets/__init__.py",
        "src/data/utils/patient_split.py",
        "src/data/utils/__init__.py",
        "src/data/__init__.py",
        # Utils
        "src/utils/checkpoint_utils.py",
        "src/utils/__init__.py",
        # Losses
        "src/losses/info_nce.py",
        "src/losses/__init__.py",
        # Scripts
        "scripts/train_pretrain.py",
        "scripts/train_finetune.py",
        "scripts/train_dual_pretrain.py",
    ]
    
    # Config files to check
    config_files = [
        "configs/config.yaml",
        "configs/config_dual.yaml",
        "configs/model/cnn_mamba.yaml",
        "configs/model/time_stream.yaml",
        "configs/model/freq_stream.yaml",
        "configs/model/dual_stream.yaml",
        "configs/model/dual_stream_wavkan.yaml",
        "configs/model/bimamba.yaml",
        "configs/model/gmu.yaml",
        "configs/augmentation/weak.yaml",
        "configs/augmentation/strong.yaml",
        "configs/augmentation/mixed.yaml",
        "configs/data/physionet2016_v2.yaml",
        "configs/data/physionet2016_dual.yaml",
        "configs/experiment/ablation_a1.yaml",
        "configs/experiment/ablation_a2.yaml",
        "configs/experiment/ablation_b1.yaml",
        "configs/experiment/ablation_b2.yaml",
        "configs/experiment/ablation_c1.yaml",
        "configs/experiment/ablation_c2.yaml",
        "configs/experiment/ablation_c3.yaml",
        "configs/logger/wandb.yaml",
    ]
    
    # Check Python syntax
    print("PYTHON SYNTAX CHECK")
    print("-" * 40)
    
    syntax_ok = 0
    syntax_fail = 0
    
    for file_path in core_files:
        is_valid, error = check_python_syntax(file_path)
        if is_valid:
            print(f"  ✓ {file_path}")
            syntax_ok += 1
        else:
            print(f"  ✗ {file_path}")
            print(f"    Error: {error}")
            syntax_fail += 1
    
    print(f"\n{syntax_ok}/{len(core_files)} Python files have valid syntax")
    
    # Check config files exist
    print("\n" + "=" * 80)
    print("CONFIG FILE CHECK")
    print("-" * 40)
    
    config_ok = 0
    config_fail = 0
    
    for file_path in config_files:
        if check_file_exists(file_path):
            print(f"  ✓ {file_path}")
            config_ok += 1
        else:
            print(f"  ✗ {file_path}")
            config_fail += 1
    
    print(f"\n{config_ok}/{len(config_files)} config files exist")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80 + "\n")
    
    total_issues = syntax_fail + config_fail
    
    if total_issues == 0:
        print("✓ All checks passed!")
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install Mamba: pip install mamba-ssm causal-conv1d")
        print("3. Run integration tests: python scripts/test_integration.py")
        print("4. Start training: python scripts/train_pretrain.py")
    else:
        print(f"✗ {total_issues} issue(s) found")
        print("Please fix the issues above before proceeding.")
    
    print()
    return total_issues == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
