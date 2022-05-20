import numpy as np

def generate_random_map(path, m, n):
    with open(path, 'w') as f:
        f.write("%d %d\n"%(m, n))
        arr = np.random.randint(0, 2, size=(m, n))
        for row in arr:
            for i in row:
                f.write("%d "%(i))
            f.write("\n")

def transform_map(path, to_path):
    arr = []
    with open(path, "r") as f:
        m, n = f.readline().strip().split(' ')
        for l in f.readlines():
            tmp = l.strip().split(' ')
            for i in tmp:
                arr.append(int(i))

    with open(to_path, "wb") as f:
        f.write((int(m)).to_bytes(4, byteorder='little'))
        f.write((int(n)).to_bytes(4, byteorder='little'))
        for i in arr:
            f.write(i.to_bytes(4, byteorder='little'))

generate_random_map('100000.txt', 100000, 100)
transform_map('100000.txt', '100000')
