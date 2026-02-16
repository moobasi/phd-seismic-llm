"""
================================================================================
MODEL WEIGHT CONVERSION UTILITIES
================================================================================

Converts pre-trained weights from various formats (Keras HDF5, TensorFlow)
to PyTorch format for use with the deep learning module.

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("h5py not available - install with: pip install h5py")


def inspect_hdf5_model(filepath: str) -> dict:
    """
    Inspect HDF5 model file structure.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary with model structure information
    """
    if not H5PY_AVAILABLE:
        return {"error": "h5py not available"}

    info = {"layers": [], "weights": {}}

    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info["weights"][name] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                }
            elif isinstance(obj, h5py.Group):
                info["layers"].append(name)

        f.visititems(visitor)

    return info


def extract_keras_weights(filepath: str) -> dict:
    """
    Extract weights from Keras HDF5 model file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary mapping layer names to weight arrays
    """
    if not H5PY_AVAILABLE:
        return {}

    weights = {}

    with h5py.File(filepath, 'r') as f:
        # Try different Keras save formats
        if 'model_weights' in f:
            weight_group = f['model_weights']
        elif 'layer_names' in f.attrs:
            weight_group = f
        else:
            weight_group = f

        def extract(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)

        weight_group.visititems(extract)

    return weights


def convert_faultseg3d_to_pytorch(keras_path: str, output_path: str) -> bool:
    """
    Convert FaultSeg3D Keras model to PyTorch state dict.

    The FaultSeg3D model is a 3D U-Net with specific layer structure.
    This function maps Keras weights to PyTorch format.

    Args:
        keras_path: Path to Keras HDF5 model
        output_path: Path to save PyTorch state dict

    Returns:
        True if successful
    """
    if not TORCH_AVAILABLE or not H5PY_AVAILABLE:
        print("PyTorch and h5py required for conversion")
        return False

    print(f"Loading Keras weights from: {keras_path}")
    keras_weights = extract_keras_weights(keras_path)

    if not keras_weights:
        print("No weights found in HDF5 file")
        return False

    print(f"Found {len(keras_weights)} weight arrays")

    # Map Keras weight names to PyTorch names
    # Keras: layer_name/kernel:0, layer_name/bias:0
    # PyTorch: layer_name.weight, layer_name.bias

    pytorch_state = {}

    for name, weight in keras_weights.items():
        # Convert name format
        pytorch_name = name.replace('/', '.').replace(':0', '')
        pytorch_name = pytorch_name.replace('kernel', 'weight')

        # Convert to tensor
        tensor = torch.from_numpy(weight)

        # Keras Conv3D: (D, H, W, in_ch, out_ch)
        # PyTorch Conv3d: (out_ch, in_ch, D, H, W)
        if 'weight' in pytorch_name and len(tensor.shape) == 5:
            tensor = tensor.permute(4, 3, 0, 1, 2)

        # Keras Dense: (in_features, out_features)
        # PyTorch Linear: (out_features, in_features)
        elif 'weight' in pytorch_name and len(tensor.shape) == 2:
            tensor = tensor.t()

        pytorch_state[pytorch_name] = tensor
        print(f"  {name} -> {pytorch_name}: {tensor.shape}")

    # Save PyTorch state dict
    torch.save(pytorch_state, output_path)
    print(f"Saved PyTorch weights to: {output_path}")

    return True


def create_weight_mapping_report(model_dir: str = "faultseg3d") -> str:
    """
    Create a report of available model weights and their structure.
    """
    report = ["=" * 70]
    report.append("MODEL WEIGHT MAPPING REPORT")
    report.append("=" * 70)

    model_path = Path(__file__).parent / model_dir

    if not model_path.exists():
        report.append(f"\nModel directory not found: {model_path}")
        return "\n".join(report)

    for hdf5_file in model_path.glob("*.hdf5"):
        report.append(f"\n\nFile: {hdf5_file.name}")
        report.append("-" * 50)

        info = inspect_hdf5_model(str(hdf5_file))

        report.append(f"Layers: {len(info.get('layers', []))}")
        report.append(f"Weight arrays: {len(info.get('weights', {}))}")

        if info.get('weights'):
            report.append("\nWeight shapes:")
            for name, details in list(info['weights'].items())[:10]:
                report.append(f"  {name}: {details['shape']} ({details['dtype']})")
            if len(info['weights']) > 10:
                report.append(f"  ... and {len(info['weights']) - 10} more")

    return "\n".join(report)


def main():
    """Main conversion routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert model weights')
    parser.add_argument('--inspect', type=str, help='Inspect HDF5 file structure')
    parser.add_argument('--convert', type=str, help='Convert Keras to PyTorch')
    parser.add_argument('--output', type=str, help='Output path for converted model')
    parser.add_argument('--report', action='store_true', help='Generate weight mapping report')

    args = parser.parse_args()

    if args.inspect:
        info = inspect_hdf5_model(args.inspect)
        print(json.dumps(info, indent=2, default=str))

    elif args.convert:
        output = args.output or args.convert.replace('.hdf5', '_pytorch.pth')
        convert_faultseg3d_to_pytorch(args.convert, output)

    elif args.report:
        print(create_weight_mapping_report())

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
