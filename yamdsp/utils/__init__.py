from math import floor, sqrt

import numpy as np
from numba import cuda
from numba import float64, float32, void, int32


def gen_dist_func(dtype, gpu):
    nb_float = float64
    if dtype == np.dtype(np.float32):
        nb_float = float32
    cuda.select_device(gpu)

    @cuda.jit(nb_float(nb_float[:], nb_float[:], nb_float[:], nb_float, nb_float), device=True)
    def cu_pbc_dist_diameter(a, b, box, da, db):
        ret = 0
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            d -= box[i] * floor(d / box[i] + 0.5)
            ret += d ** 2
        return sqrt(ret) - ((da + db) / 2 - 1)

    @cuda.jit(nb_float(nb_float[:], nb_float[:], nb_float[:]), device=True)
    def cu_pbc_dist2(a, b, box):
        ret = 0
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            d -= box[i] * floor(d / box[i] + 0.5)
            ret += d ** 2
        return ret

    @cuda.jit(nb_float(nb_float[:], nb_float[:], nb_float[:]), device=True)
    def cu_pbc_dist(a, b, box):
        ret = 0
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            d -= box[i] * floor(d / box[i] + 0.5)
            ret += d ** 2
        return sqrt(ret)

    return cu_pbc_dist2, cu_pbc_dist_diameter, cu_pbc_dist


@cuda.jit(int32(int32[:], int32[:]), device=True)
def cu_ravel_index_f_pbc(i, dim):  # ravel index in Fortran way.
    ret = (i[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, dim.shape[0]):
        ret += ((i[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@cuda.jit(void(int32, int32[:], int32[:]), device=True)
def cu_unravel_index_f(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


__all__ = ["cu_ravel_index_f_pbc",
           "cu_unravel_index_f",
           "gen_func"
           ]
