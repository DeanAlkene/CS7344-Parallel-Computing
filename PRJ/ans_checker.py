import numpy as np
import torch

def wordcount_checker(path, path_gt):
    d = {}
    with open(path_gt, "rb") as f:
        ls = f.readlines()
        for l in ls:
            sl = l.split(b' ')
            if d.get(sl[0]) is not None:
                raise KeyError("Duplicate key!")
            d[sl[0]] = int(int(sl[1].strip()))
    
    with open(path, "rb") as f:
        ls = f.readlines()
        for l in ls:
            sl = l.split(b' ')
            val = d.get(sl[0])
            if val is None:
                raise KeyError("%s not exist!"%(sl[0]))
            if val != int(int(sl[1].strip())):
                raise ValueError("%s has wrong count!"%(sl[0]))
            d.pop(sl[0])
    print("Success!")

def wordcount_checker_shuffle(name, path_gt, reduce_p):
    d = {}
    with open(path_gt, "rb") as f:
        ls = f.readlines()
        for l in ls:
            sl = l.split(b' ')
            if d.get(sl[0]) is not None:
                raise KeyError("Duplicate key!")
            d[sl[0]] = int(int(sl[1].strip()))
    
    d_res = {}
    ids = [i for i in range(reduce_p)]
    for i in ids:
        with open(name + "_" + str(i) + ".txt", "rb") as f:
            ls = f.readlines()
            for l in ls:
                sl = l.split(b' ')
                if d_res.get(sl[0]) is not None:
                    raise KeyError("Duplicate key!")
                d_res[sl[0]] = int(int(sl[1].strip()))
    
    for key in d_res.keys():
        val = d.get(key)
        if val is None:
            raise KeyError("%s not exist!"%(key))
        if val != d_res[key]:
            raise ValueError("%s has wrong count!"%(key))
        d.pop(key)
        
    if len(d) != 0:
        raise ValueError("Key numbers are not equal")
    # if d != d_res:
    #     raise ValueError("Dicts are not equal")
    
    print("Success!")

def gemm_checker(path_A, path_B, path_C, dtype):
    with open(path_A, "rb") as f:
        m_A = int.from_bytes(f.read(4), byteorder='little')
        n_A = int.from_bytes(f.read(4), byteorder='little')
        A = np.frombuffer(f.read(m_A * n_A * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_A, n_A))
    
    with open(path_B, "rb") as f:
        m_B = int.from_bytes(f.read(4), byteorder='little')
        n_B = int.from_bytes(f.read(4), byteorder='little')
        B = np.frombuffer(f.read(m_B * n_B * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_B, n_B))
    
    with open(path_C, "rb") as f:
        m_C = int.from_bytes(f.read(4), byteorder='little')
        n_C = int.from_bytes(f.read(4), byteorder='little')
        C = np.frombuffer(f.read(m_C * n_C * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_C, n_C))
    
    C_gt = np.matmul(A, B)
    if dtype == np.int32:
        delta = (C == C_gt)
    else:
        delta = (C - C_gt) < 1e-9
        
    if not delta.all():
        raise ValueError("Wrong result")
    
    print("Success!")
        
def conv_checker(path_f, path_k, path_C, stride, dtype):
    with open(path_f, "rb") as f:
        m_A = int.from_bytes(f.read(4), byteorder='little')
        n_A = int.from_bytes(f.read(4), byteorder='little')
        A = np.frombuffer(f.read(m_A * n_A * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_A, n_A))
    
    with open(path_k, "rb") as f:
        m_B = int.from_bytes(f.read(4), byteorder='little')
        n_B = int.from_bytes(f.read(4), byteorder='little')
        B = np.frombuffer(f.read(m_B * n_B * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_B, n_B))
    
    with open(path_C, "rb") as f:
        m_C = int.from_bytes(f.read(4), byteorder='little')
        n_C = int.from_bytes(f.read(4), byteorder='little')
        C = np.frombuffer(f.read(m_C * n_C * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_C, n_C))
    
    cmp_dtype = np.float32 if dtype != np.float64 else np.float64
    A = torch.from_numpy(A.astype(cmp_dtype)).unsqueeze(0).unsqueeze(0)
    B = torch.from_numpy(B.astype(cmp_dtype)).unsqueeze(0).unsqueeze(0)
    conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[m_B, n_B], stride=stride, bias=False)
    conv2d.requires_grad_(False)
    conv2d.weight = torch.nn.Parameter(B, requires_grad=False)
    
    C_gt = conv2d(A).squeeze(0).squeeze(0)
    if C_gt.shape[0] != m_C or C_gt.shape[1] != n_C:
        raise ValueError("Wrong result shape")

    delta = np.abs((C.astype(cmp_dtype) - C_gt.numpy())) < 1e-9
        
    if not delta.all():
        raise ValueError("Wrong result")
    
    print("Success!")

def avgpooling_checker(path_f, path_C, kernel_size, stride, dtype):
    with open(path_f, "rb") as f:
        m_A = int.from_bytes(f.read(4), byteorder='little')
        n_A = int.from_bytes(f.read(4), byteorder='little')
        A = np.frombuffer(f.read(m_A * n_A * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_A, n_A))
    
    with open(path_C, "rb") as f:
        m_C = int.from_bytes(f.read(4), byteorder='little')
        n_C = int.from_bytes(f.read(4), byteorder='little')
        C = np.frombuffer(f.read(m_C * n_C * np.dtype(dtype).itemsize), dtype=dtype).reshape((m_C, n_C))
    
    cmp_dtype = np.float32 if dtype != np.float64 else np.float64
    A = torch.from_numpy(A.astype(cmp_dtype)).unsqueeze(0).unsqueeze(0)
    avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    avgpool.requires_grad_(False)
    
    C_gt = avgpool(A).squeeze(0).squeeze(0)
    if C_gt.shape[0] != m_C or C_gt.shape[1] != n_C:
        raise ValueError("Wrong result shape")

    delta = np.abs((C.astype(cmp_dtype) - C_gt.numpy())) < 1e-9
        
    if not delta.all():
        raise ValueError("Wrong result")
    
    print("Success!")

# wordcount_checker("wordcount_small.txt", "wordcount_small_gt.txt")
# wordcount_checker_shuffle("wordcount_big", "wordcount_big_gt.txt", 4)
gemm_checker("mat_A", "mat_B", "C", np.float64)
conv_checker("mat_A", "kernel", "Conv", [1, 1], np.float64)
avgpooling_checker("mat_A", "AvgPooling", [4, 4], [1, 1], np.float64)