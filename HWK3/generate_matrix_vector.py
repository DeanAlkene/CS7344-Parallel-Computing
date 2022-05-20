import numpy as np

def generate_random_matrix_vector(m, n, path_m, path_v):
    mat = np.random.random((m, n))
    vec = np.random.random((n))

    with open(path_m, "wb") as f:
        f.write((int(m)).to_bytes(4, byteorder='little'))
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write(mat.tobytes())
    
    with open(path_v, "wb") as f:
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write(vec.tobytes())

def generate_prob_vector(n, path):
    # vec = np.array([0.16, 0.13, 0.06, 0.08, 0.07, 0.17, 0.05, 0.28], dtype=np.float32)
    vec = np.random.random((n))
    vec = vec / np.sum(vec)
    
    with open(path, "wb") as f:
        f.write((int(n)).to_bytes(4, byteorder='little'))
        f.write(vec.tobytes())
    
    with open(path + '.txt', "w") as f:
        f.write("%d\n"%(n))
        for i in vec:
            f.write("%0.2f "%(i))

generate_random_matrix_vector(100, 100, 'mat', 'vec')
# generate_prob_vector(50, 'prob')