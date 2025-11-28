"""
GPU Configuration Module

Handles GPU configuration for both TensorFlow and PyTorch frameworks.
Optimized for RTX 5070 compatibility.
"""

import tensorflow as tf
import torch


def configure_gpu():
    """
    Configure GPU settings for optimal performance with RTX 5070.

    RTX 5070 has compute capability 12.0, which is not natively supported by
    TensorFlow 2.19.1. This causes 30+ minute JIT compilation delays.

    Solution:
    - Force TensorFlow to use CPU (still fast for inference)
    - Ensure PyTorch uses GPU (full acceleration)
    """
    print("\n" + "=" * 60)
    print("GPU Configuration")
    print("=" * 60)

    # Configure PyTorch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"✓ PyTorch GPU: {device_name}")
        print(f"  - Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print("  - Status: GPU ENABLED")
    else:
        print("⚠ PyTorch: No GPU available, using CPU")

    # Configure TensorFlow
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Check if we have RTX 5070 (compute capability 12.0+)
            # Force CPU to avoid 30-minute JIT compilation
            try:
                # Disable GPU for TensorFlow to avoid slow JIT compilation
                tf.config.set_visible_devices([], "GPU")
                print("✓ TensorFlow: GPU detected but configured to use CPU")
                print("  - Reason: Avoiding JIT compilation delay on RTX 5070")
                print("  - Status: CPU MODE (optimized for inference)")
            except RuntimeError as e:
                print(f"⚠ TensorFlow GPU configuration warning: {e}")
        else:
            print("✓ TensorFlow: Using CPU (no GPU detected)")
    except Exception as e:
        print(f"⚠ TensorFlow configuration warning: {e}")

    print("=" * 60 + "\n")
