import sys
import numpy as np


def inspect_npz(path):
    data = np.load(path)
    print(f"File: {path}")
    print(f"Keys: {list(data.keys())}\n")

    for key in data.keys():
        arr = data[key]
        print(f"--- {key} ---")
        print(f"  shape: {arr.shape}   dtype: {arr.dtype}")
        if arr.ndim >= 1:
            print(f"  rows (dim0): {arr.shape[0]}")
        if arr.ndim >= 2:
            print(f"  cols (dim1): {arr.shape[1]}")
        print(f"  first 3 rows:\n{arr[:3]}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data_traj/path_42_15_10_15.npz"
    inspect_npz(path)
