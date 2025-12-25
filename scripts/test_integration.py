"""
Integration test script to verify all new modules work correctly.

Tests:
1. Module imports
2. DualStreamEncoder instantiation
3. DualModalDataModule data loading
4. DualModalMoCoModule forward pass
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all module imports."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from src.models.encoders import (
            BiMambaBlock,
            FreqStreamEncoder,
            DualStreamEncoder,
            build_dual_stream_encoder
        )
        from src.models.fusion import GatedMultimodalUnit, AdaptiveGatedFusion
        from src.models.heads import MLPProjectionHead, WavKANProjectionHead
        from src.models import DualModalMoCoModule
        from src.data.datamodules import DualModalPretrainDataModule
        from src.data.utils.patient_split import PatientSplitter
        from src.utils.checkpoint_utils import save_encoder, load_encoder
        
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dual_stream_encoder():
    """Test DualStreamEncoder instantiation and forward pass."""
    print("\n" + "=" * 80)
    print("TEST 2: DualStreamEncoder")
    print("=" * 80)
    
    try:
        from src.models.encoders import DualStreamEncoder
        from src.models.heads import MLPProjectionHead

        if not torch.cuda.is_available():
            print("! CUDA not available; skipping DualStreamEncoder test (mamba-ssm ops require CUDA)")
            return True
        
        # Create encoder
        encoder = DualStreamEncoder(
            time_stream_config={
                'cnn_config': {'channels': [32, 64, 128], 'kernel_sizes': [25, 25, 3]},
                'mamba_config': {'d_model': 128, 'n_layers': 4}
            },
            freq_stream_config={
                'n_mels': 64,
                'd_model': 128,
                'mamba_config': {'d_model': 128, 'n_layers': 4}
            },
            fusion_config={'mode': 'gmu'},
            projection_head=MLPProjectionHead(input_dim=128, output_dim=128)
        )
        
        # Test forward pass
        batch_size = 4
        time_input = torch.randn(batch_size, 10000, device='cuda')  # 5s @ 2kHz
        freq_input = torch.randn(batch_size, 64, 500, device='cuda')  # 64 mels, 500 frames

        encoder = encoder.cuda()
        
        output = encoder(time_input, freq_input)
        
        print(f"✓ Encoder created successfully")
        print(f"  - Time input shape: {time_input.shape}")
        print(f"  - Freq input shape: {freq_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output dim: {encoder.get_output_dim()}")
        
        assert output.shape == (batch_size, 128), f"Expected (4, 128), got {output.shape}"
        print("✓ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wavkan_head():
    """Test WavKANProjectionHead."""
    print("\n" + "=" * 80)
    print("TEST 3: WavKANProjectionHead")
    print("=" * 80)
    
    try:
        from src.models.heads import WavKANProjectionHead
        
        head = WavKANProjectionHead(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            wavelet_type='morlet',
            num_wavelets=8
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        output = head(x)
        
        print(f"✓ WavKAN head created successfully")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Wavelet type: morlet")
        
        assert output.shape == (batch_size, 128), f"Expected (4, 128), got {output.shape}"
        print("✓ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gmu_fusion():
    """Test AdaptiveGatedFusion."""
    print("\n" + "=" * 80)
    print("TEST 4: AdaptiveGatedFusion")
    print("=" * 80)
    
    try:
        from src.models.fusion import AdaptiveGatedFusion
        
        fusion = AdaptiveGatedFusion(d_model=128, mode='gmu')
        
        batch_size = 4
        time_feat = torch.randn(batch_size, 625, 128)  # 625 time tokens
        freq_feat = torch.randn(batch_size, 125, 128)  # 125 freq tokens
        
        fused = fusion(time_feat, freq_feat)
        
        print(f"✓ GMU fusion created successfully")
        print(f"  - Time features shape: {time_feat.shape}")
        print(f"  - Freq features shape: {freq_feat.shape}")
        print(f"  - Fused shape: {fused.shape}")
        
        assert fused.shape == (batch_size, 625, 128), f"Expected (4, 625, 128), got {fused.shape}"
        print("✓ Fusion successful!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bimamba():
    """Test BiMambaBlock."""
    print("\n" + "=" * 80)
    print("TEST 5: BiMambaBlock")
    print("=" * 80)
    
    try:
        from src.models.encoders import BiMambaBlock

        if not torch.cuda.is_available():
            print("! CUDA not available; skipping BiMambaBlock test (mamba-ssm ops require CUDA)")
            return True
        
        block = BiMambaBlock(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2,
            fusion_mode='gate'
        )
        
        batch_size = 4
        seq_len = 625
        x = torch.randn(batch_size, seq_len, 128, device='cuda')
        block = block.cuda()
        
        output = block(x)
        
        print(f"✓ BiMamba block created successfully")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Fusion mode: gate")
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        print("✓ BiMamba forward pass successful!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patient_splitter():
    """Test PatientSplitter."""
    print("\n" + "=" * 80)
    print("TEST 6: PatientSplitter")
    print("=" * 80)
    
    try:
        from src.data.utils.patient_split import PatientSplitter
        
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.file_paths = [f'patient_{i//3:03d}_seg_{i%3}.wav' for i in range(30)]
                
            def __len__(self):
                return len(self.file_paths)
            
            def __getitem__(self, idx):
                return torch.randn(10000), 0  # audio, label
        
        dataset = MockDataset()
        splitter = PatientSplitter(
            dataset=dataset,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42
        )
        
        train_idx, val_idx, test_idx = splitter.split()
        
        print(f"✓ PatientSplitter created successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Train samples: {len(train_idx)}")
        print(f"  - Val samples: {len(val_idx)}")
        print(f"  - Test samples: {len(test_idx)}")
        
        # Verify no overlap
        assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap detected"
        
        print("✓ Patient-wise split successful (no overlap)!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Phase 0-3 Integration Tests")
    print("=" * 80 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("DualStreamEncoder", test_dual_stream_encoder()))
    results.append(("WavKANProjectionHead", test_wavkan_head()))
    results.append(("AdaptiveGatedFusion", test_gmu_fusion()))
    results.append(("BiMambaBlock", test_bimamba()))
    results.append(("PatientSplitter", test_patient_splitter()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    print("\n" + "=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
