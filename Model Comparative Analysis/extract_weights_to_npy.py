"""
Script to extract weights from a corrupted .weights.h5 file and save as .npy file
"""
import h5py
import numpy as np
import os
from pathlib import Path

def extract_weights_from_h5(h5_file_path, output_dir=None):
    """
    Extract weights from a .weights.h5 file and save as .npy files
    
    Args:
        h5_file_path: Path to the .weights.h5 file
        output_dir: Directory to save the .npy files (default: same as h5 file)
    """
    h5_file_path = Path(h5_file_path)
    
    if not h5_file_path.exists():
        raise FileNotFoundError(f"File not found: {h5_file_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = h5_file_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base name for output files
    base_name = h5_file_path.stem.replace('.weights', '')
    
    print(f"Reading weights from: {h5_file_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Open the HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            print(f"\nFile structure:")
            print_h5_structure(f)
            
            # Extract all weights
            all_weights = {}
            extract_weights_recursive(f, all_weights, prefix='')
            
            if not all_weights:
                print("\nWarning: No weights found in the file!")
                return
            
            print(f"\nFound {len(all_weights)} weight arrays")
            
            # Save individual weight arrays
            for name, weight in all_weights.items():
                # Clean the name for filename
                clean_name = name.replace('/', '_').strip('_')
                output_file = output_dir / f"{base_name}_{clean_name}.npy"
                np.save(output_file, weight)
                print(f"Saved: {output_file.name} - Shape: {weight.shape}, dtype: {weight.dtype}")
            
            # Save all weights as a single dictionary
            combined_file = output_dir / f"{base_name}_all_weights.npy"
            np.save(combined_file, all_weights)
            print(f"\nSaved combined weights to: {combined_file.name}")
            
            # Also save as a list (in layer order)
            weights_list = list(all_weights.values())
            list_file = output_dir / f"{base_name}_weights_list.npy"
            np.save(list_file, weights_list, allow_pickle=True)
            print(f"Saved weights list to: {list_file.name}")
            
            print(f"\n✓ Successfully extracted {len(all_weights)} weight arrays!")
            
    except Exception as e:
        print(f"\n✗ Error reading file: {e}")
        print("\nAttempting alternative extraction method...")
        try:
            extract_weights_alternative(h5_file_path, output_dir, base_name)
        except Exception as e2:
            print(f"✗ Alternative method also failed: {e2}")
            raise

def print_h5_structure(h5_group, indent=0):
    """Recursively print the structure of an HDF5 file"""
    for key in h5_group.keys():
        item = h5_group[key]
        print("  " * indent + f"├─ {key}", end='')
        if isinstance(item, h5py.Group):
            print(" (Group)")
            print_h5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f" (Dataset) - Shape: {item.shape}, dtype: {item.dtype}")

def extract_weights_recursive(h5_group, weights_dict, prefix=''):
    """Recursively extract all datasets (weights) from HDF5 file"""
    for key in h5_group.keys():
        item = h5_group[key]
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            # Recursively process groups
            extract_weights_recursive(item, weights_dict, full_key)
        elif isinstance(item, h5py.Dataset):
            # Extract dataset (weight array)
            weights_dict[full_key] = item[:]

def extract_weights_alternative(h5_file_path, output_dir, base_name):
    """
    Alternative method to extract weights using lower-level HDF5 access
    This might work even if the file is partially corrupted
    """
    import h5py
    
    print("Trying alternative extraction method...")
    
    with h5py.File(h5_file_path, 'r') as f:
        # Try to access common weight storage locations
        possible_paths = [
            'model_weights',
            'weights',
            '',  # root level
        ]
        
        weights_found = False
        for path in possible_paths:
            try:
                if path:
                    group = f[path]
                else:
                    group = f
                
                # Try to extract from this location
                all_weights = {}
                extract_weights_recursive(group, all_weights, prefix='')
                
                if all_weights:
                    weights_found = True
                    print(f"Found weights at path: '{path}'")
                    
                    # Save the weights
                    for i, (name, weight) in enumerate(all_weights.items()):
                        clean_name = name.replace('/', '_').strip('_')
                        output_file = output_dir / f"{base_name}_alt_{clean_name}.npy"
                        np.save(output_file, weight)
                        print(f"Saved: {output_file.name} - Shape: {weight.shape}")
                    
                    break
            except:
                continue
        
        if not weights_found:
            raise ValueError("Could not find weights in any expected location")

def main():
    """Main function with example usage"""
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument
        h5_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Interactive mode
        print("=" * 60)
        print("H5 Weights to NPY Converter")
        print("=" * 60)
        
        # Look for .weights.h5 files in current directory
        current_dir = Path('.')
        h5_files = list(current_dir.glob('*.weights.h5')) + list(current_dir.glob('*.h5'))
        
        if h5_files:
            print("\nFound .h5 files in current directory:")
            for i, f in enumerate(h5_files, 1):
                print(f"  {i}. {f.name}")
            print(f"  {len(h5_files) + 1}. Enter custom path")
            
            choice = input(f"\nSelect file (1-{len(h5_files) + 1}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(h5_files):
                    h5_file = str(h5_files[choice_idx])
                else:
                    h5_file = input("Enter path to .weights.h5 file: ").strip()
            except ValueError:
                h5_file = input("Enter path to .weights.h5 file: ").strip()
        else:
            h5_file = input("Enter path to .weights.h5 file: ").strip()
        
        output_dir = input("Enter output directory (press Enter for same directory): ").strip()
        output_dir = output_dir if output_dir else None
    
    # Remove quotes if present
    h5_file = h5_file.strip('"').strip("'")
    if output_dir:
        output_dir = output_dir.strip('"').strip("'")
    
    # Extract weights
    extract_weights_from_h5(h5_file, output_dir)

if __name__ == "__main__":
    main()
