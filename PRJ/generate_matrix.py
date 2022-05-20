import numpy as np

def generate_random_matrix(m, n, path_m, dtype):
    if dtype != np.int32:
        mat = np.random.random((m, n)).astype(dtype)
    else:
        mat = np.random.randint(low=0, high=128, size=(m, n), dtype=dtype)

    with open(path_m, "wb") as f:
        f.write((int(m)).to_bytes(4, byteorder='little'))
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write(mat.tobytes())
    
    return mat

def generate_kernel(n, path, dtype):
    if dtype != np.int32:
        kernel = np.random.random((n, n)).astype(dtype)
    else:
        kernel = np.random.randint(low=0, high=128, size=(n, n), dtype=dtype)
    
    with open(path, "wb") as f:
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write(kernel.tobytes())
    
    return kernel

A = generate_random_matrix(1024, 1024, "mat_A", dtype=np.float64)
B = generate_random_matrix(1024, 1024, "mat_B", dtype=np.float64)
kernel = generate_kernel(4, "kernel", dtype=np.float64)