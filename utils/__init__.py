from math import floor, sqrt

from numba import cuda


@cuda.jit("float64(float64[:], float64[:], float64[:])", device=True)
def cu_pbc_dist(a, b, box):
    ret = 0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        d -= box[i] * floor(d / box[i] + 0.5)
        ret += d ** 2
    return sqrt(ret)


@cuda.jit("float64(float64[:], float64[:], float64[:], float64, float64)", device=True)
def cu_pbc_dist_diameter(a, b, box, da, db):
    ret = 0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        d -= box[i] * floor(d / box[i] + 0.5)
        ret += d ** 2
    return sqrt(ret) - ((da + db) / 2 - 1)


@cuda.jit("float64(float64[:], float64[:], float64[:])", device=True)
def cu_pbc_dist2(a, b, box):
    ret = 0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        d -= box[i] * floor(d / box[i] + 0.5)
        ret += d ** 2
    return ret


@cuda.jit("void(int32, int32[:], int32[:])", device=True)
def cu_unravel_index_f(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


@cuda.jit("int32(int32[:], int32[:])", device=True)
def cu_ravel_index_f_pbc(i, dim):  # ravel index in Fortran way.
    ret = (i[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, dim.shape[0]):
        ret += ((i[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret
