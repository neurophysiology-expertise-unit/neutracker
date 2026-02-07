
import h5py
import numpy as np
import sys
import os

def compare_h5(file1, file2):
    print(f"Comparing {file1} vs {file2}")
    if not os.path.exists(file1):
        print(f"File not found: {file1}")
        return False
    if not os.path.exists(file2):
        print(f"File not found: {file2}")
        return False
        
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        return compare_groups(f1, f2)

def compare_groups(g1, g2, path="/"):
    keys1 = set(g1.keys())
    keys2 = set(g2.keys())
    
    if keys1 != keys2:
        print(f"Keys mismatch at {path}")
        print(f"In 1 but not 2: {keys1 - keys2}")
        print(f"In 2 but not 1: {keys2 - keys1}")
        return False
        
    for key in keys1:
        item1 = g1[key]
        item2 = g2[key]
        item_path = f"{path}{key}"
        
        if isinstance(item1, h5py.Group) and isinstance(item2, h5py.Group):
            if not compare_groups(item1, item2, item_path + "/"):
                return False
        elif isinstance(item1, h5py.Dataset) and isinstance(item2, h5py.Dataset):
            if not compare_datasets(item1, item2, item_path):
                return False
        else:
            print(f"Type mismatch at {item_path}: {type(item1)} vs {type(item2)}")
            return False
            
    return True

def compare_datasets(d1, d2, path):
    if d1.shape != d2.shape:
        print(f"Shape mismatch at {path}: {d1.shape} vs {d2.shape}")
        return False
        
    if d1.dtype != d2.dtype:
        print(f"Dtype mismatch at {path}: {d1.dtype} vs {d2.dtype}")
        # Continue if just endianness vs native
    
    try:
        val1 = d1[:]
        val2 = d2[:]
        
        if np.issubdtype(d1.dtype, np.floating):
            if not np.allclose(val1, val2, equal_nan=True):
                diff = np.abs(val1 - val2)
                max_diff = np.nanmax(diff)
                print(f"Value mismatch at {path}. Max Diff: {max_diff}")
                return False
        else:
            if not np.array_equal(val1, val2):
                print(f"Value mismatch at {path} (exact match required)")
                return False
    except Exception as e:
        print(f"Error comparing data at {path}: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_hdf5.py file1.hdf5 file2.hdf5")
        sys.exit(1)
        
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    
    if compare_h5(f1, f2):
        print("Files match!")
        sys.exit(0)
    else:
        print("Files differ!")
        sys.exit(1)
