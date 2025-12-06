"""
Test script to verify that all components are working correctly
Run this before running full experiments
"""
import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    errors = []
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import tensorflow as tf
        print(f"✓ tensorflow {tf.__version__}")
    except ImportError as e:
        errors.append(f"✗ tensorflow: {e}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {e}")
    
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
        print("✓ tensorflow_privacy")
    except ImportError as e:
        errors.append(f"✗ tensorflow_privacy: {e}")
    
    try:
        from dp_accounting import dp_event
        print("✓ dp_accounting")
    except ImportError as e:
        errors.append(f"✗ dp_accounting: {e}")
    
    return errors


def test_composite_dp():
    """Test that CompositeDP can be imported and used"""
    print("\nTesting CompositeDP...")
    errors = []
    
    try:
        # Add CompositeDP to path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        COMPOSITE_DP_DIR = os.path.join(BASE_DIR, "CompositeDP")
        if COMPOSITE_DP_DIR not in sys.path:
            sys.path.insert(0, COMPOSITE_DP_DIR)
        
        from CompositeDP.Perturbation_Mechanism import perturbation_fun_oneCall
        print("✓ CompositeDP imports successfully")
        
        # Test basic functionality
        epsilon = 1.0
        fd = 30
        sensitivity = 10
        lower_bound = 0
        k = 0.5
        m = 0.4
        y = 0.3
        index = 1
        
        result = perturbation_fun_oneCall(epsilon, fd, sensitivity, lower_bound, k, m, y, index)
        print(f"✓ CompositeDP generates noise: {result:.4f}")
        
    except ImportError as e:
        errors.append(f"✗ CompositeDP import failed: {e}")
    except Exception as e:
        errors.append(f"✗ CompositeDP function failed: {e}")
    
    return errors


def test_datasets():
    """Test that datasets can be loaded"""
    print("\nTesting datasets...")
    errors = []
    
    try:
        from datasets import load_mnist
        x_train, y_train, x_val, y_val, x_test, y_test, meta = load_mnist()
        print(f"✓ MNIST loaded: {x_train.shape[0]} train samples")
    except Exception as e:
        errors.append(f"✗ MNIST loading failed: {e}")
    
    try:
        from datasets import load_adult
        x_train, y_train, x_val, y_val, x_test, y_test, meta = load_adult()
        print(f"✓ Adult loaded: {x_train.shape[0]} train samples")
    except Exception as e:
        errors.append(f"✗ Adult loading failed: {e}")
    
    return errors


def test_models():
    """Test that models can be built"""
    print("\nTesting models...")
    errors = []
    
    try:
        import tensorflow as tf
        from models import build_mnist_cnn
        model = build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10)
        print(f"✓ MNIST CNN built: {model.count_params()} parameters")
    except Exception as e:
        errors.append(f"✗ MNIST CNN failed: {e}")
    
    try:
        from models import build_adult_mlp
        model = build_adult_mlp(input_dim=100, num_classes=2)
        print(f"✓ Adult MLP built: {model.count_params()} parameters")
    except Exception as e:
        errors.append(f"✗ Adult MLP failed: {e}")
    
    return errors


def test_composite_dp_utils():
    """Test composite_dp_utils functions"""
    print("\nTesting composite_dp_utils...")
    errors = []
    
    try:
        from composite_dp_utils import add_composite_dp_noise_to_gradient
        import numpy as np
        
        # Test noise generation
        gradient = np.ones((10,), dtype=np.float32)
        noisy_gradient = add_composite_dp_noise_to_gradient(
            gradient_array=gradient,
            epsilon=1.0,
            sensitivity=1.0,
            lower_bound=0.0,
            k=0.5,
            m=0.4,
            y=0.3,
            index=1,
        )
        print(f"✓ Noise added to gradient: mean={np.mean(noisy_gradient):.4f}")
        
    except Exception as e:
        errors.append(f"✗ composite_dp_utils failed: {e}")
    
    return errors


def main():
    print("="*60)
    print("SGD_CDP_Project Setup Test")
    print("="*60)
    
    all_errors = []
    
    # Run all tests
    all_errors.extend(test_imports())
    all_errors.extend(test_composite_dp())
    all_errors.extend(test_datasets())
    all_errors.extend(test_models())
    all_errors.extend(test_composite_dp_utils())
    
    print("\n" + "="*60)
    if all_errors:
        print("TESTS FAILED")
        print("="*60)
        print("\nErrors found:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix these errors before running experiments.")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nYour setup is ready! You can now run experiments:")
        print("  python train_experiment_tf.py --dataset mnist --mechanism none --epochs 1 --batch_size 256")
        sys.exit(0)


if __name__ == "__main__":
    main()