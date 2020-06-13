import numpy as np
from numba import cuda, float64, float32, int32, void

from .._helpers import Ctx


class pair:
    n_types: int

    def __init__(self, name, r_cut):
        self.name = name
        self.r_cut = r_cut
        system = Ctx.get_active()
        if system is not None:
            self.types = system.types
            self.typeid = system.typeid
            assert isinstance(self.types, list)
            self.n_types = len(self.types)
            self.nlist = system.nlist
            self.system = system
        else:
            raise ValueError("No active system, initialize system first!")
        self.params = [[] for _ in range(int(self.n_types * self.n_types))]

    def set_params(self, type_a, type_b, *args):
        tid_a: int = self.types.index(type_a)
        tid_b: int = self.types.index(type_b)
        self.params[tid_a * self.n_types + tid_b] = list(args)
        self.params[tid_b * self.n_types + tid_a] = list(args)
        if np.equal.reduce([len(_) for _ in self.params]):
            self.params = np.asarray(self.params, dtype=np.float64)

    def get_params(self):
        return self.params

    def check_params(self):
        return isinstance(self.params, np.ndarray)

    def force_functions(self, funcs):  # general pair cases
        # @cuda.jit("void(float64[:], float64[:], float64[:], float64[:], float64[:,:])", device=True)
        # def func(a, b, param, forces):
        #    pass
        nb_float = float64
        if self.system.dtype == np.dtype(np.float32):
            nb_float = float32
        kernels = []
        cu_pbc_dist2 = self.nlist.dist_funcs['cu_pbc_dist2']
        for f in funcs:
            @cuda.jit(void(nb_float[:,:], nb_float[:], int32[:], int32[:], nb_float[:], int32[:], int32, nb_float[:]))
            def _f(x, box, nl, nc, params, typeid, n_types, forces):
                i = cuda.grid(1)
                if i >= x.shape[0]:
                    return
                xi = x[i]
                ti = typeid[i]
                for k in range(nc[i]):
                    j = nl[i, k]
                    tj = typeid[j]
                    dij2 = cu_pbc_dist2(xi, x[j], box)
                    f(dij2, box, params[ti * n_types + tj], forces)

            kernels.append(_f)
        return kernels
