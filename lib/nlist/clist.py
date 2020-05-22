from math import floor

import numpy as np
from numba import cuda
from numba import int32, float32, float64, void

from . import cu_set_to_int
from .._helpers import Ctx
from ..utils import cu_unravel_index_f, cu_ravel_index_f_pbc


def _gen_func(dtype, n_dim):
    float = float64
    if dtype == np.dtype(np.float32):
        float = float32

    @cuda.jit(int32(float[:], float[:], int32[:]), device=True)
    def cu_cell_index(x, box, ibox):
        ret = floor((x[0] / box[0] + 0.5) * ibox[0])
        n_cell = ibox[0]
        for i in range(1, x.shape[0]):
            ret = ret + floor((x[i] / box[i] + 0.5) * ibox[i]) * n_cell
            n_cell = n_cell * ibox[i]
        return ret

    @cuda.jit(void(int32[:], int32[:], int32[:, :]))
    def cu_cell_map(ibox, dim, ret):
        cell_i = cuda.grid(1)
        if cell_i >= ret.shape[0]:
            return
        cell_vec_i = cuda.local.array(n_dim, int32)
        cell_vec_j = cuda.local.array(n_dim, int32)
        cu_unravel_index_f(cell_i, ibox, cell_vec_i)
        for j in range(ret.shape[1]):
            cu_unravel_index_f(j, dim, cell_vec_j)
            for k in range(n_dim):
                cell_vec_j[k] = cell_vec_i[k] + cell_vec_j[k] - 1
            cell_j = cu_ravel_index_f_pbc(cell_vec_j, ibox)
            ret[cell_i, j] = cell_j

    @cuda.jit(void(float[:, :], float[:], int32[:], float[:, :, :], int32[:, :], int32[:], int32[:], int32[:]))
    def cu_cell_list(x, box, ibox, cell_list, cell_list_index, cell_counts, cells, cell_max):
        pi = cuda.grid(1)
        if pi >= x.shape[0]:
            return
        # xi = cuda.local.array(ndim, dtype=float64)
        # for k in range(ndim):
        # xi[k] = x[pi, k]
        xi = x[pi]
        ic = cu_cell_index(xi, box, ibox)
        cells[pi] = ic
        index = cuda.atomic.add(cell_counts, ic, 1)
        if index < cell_list.shape[0]:
            for k in range(n_dim):
                cell_list[ic, index, k] = xi[k]
            cell_list_index[ic, index] = pi
        else:
            cuda.atomic.max(cell_max, 0, index + 1)

    return cu_cell_index, cu_cell_map, cu_cell_list


class clist:
    def __init__(self, r_cut, r_buff=0.5, cell_guess=50):
        system = Ctx.get_active()
        if system is None:
            raise ValueError("Error, Initialize system first!")
        self.system = system
        self.ibox = np.asarray(np.floor(system.box / (r_cut + r_buff)), dtype=np.int32)
        self.n_cell = int(np.multiply.reduce(self.ibox))
        self.cell_adj = np.ones(self.system.n_dim, dtype=np.int32) * 3
        self.gpu = system.gpu
        self.tpb = 64
        self.bpg = int(self.system.N // self.tpb + 1)
        self.bpg_cell = int(self.n_cell // self.tpb + 1)
        self.cell_guess = cell_guess
        # self.situ_zero = np.zeros(1, dtype=np.int32)
        global cu_cell_index, cu_cell_map, cu_cell_list
        cu_cell_index, cu_cell_map, cu_cell_list = _gen_func(system.dtype, system.n_dim)
        self.p_cell_max = cuda.pinned_array((1,), dtype=np.int32)
        with cuda.gpus[self.gpu]:
            self.d_last_x = cuda.device_array_like(self.system.d_x)
            self.d_cells = cuda.device_array(self.system.d_x.shape[0], dtype=np.int32)
            self.d_cell_map = cuda.device_array((self.n_cell, 3 ** system.n_dim), dtype=np.int32)
            self.d_ibox = cuda.to_device(self.ibox)
            self.d_cell_adj = cuda.to_device(self.cell_adj)
            cu_cell_map[self.bpg_cell, self.tpb](self.d_ibox, self.d_cell_adj, self.d_cell_map)
            self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess, self.system.n_dim), dtype=np.int32)
            self.d_cell_list_index = cuda.device_array((self.ncell, self.cell_guess), dtype=np.int32)
            self.d_cell_counts = cuda.device_array(self.n_cell, dtype=np.int32)
            self.p_cell_max = cuda.pinned_array(1, dtype=np.int32)
            self.d_cell_max = cuda.device_array(1, dtype=np.int32)
        self.update()

    def update(self):
        with cuda.gpus[self.gpu]:
            while True:
                cu_set_to_int[self.bpg_cell, self.tpb](self.d_cell_counts, 0)
                cu_cell_list[self.bpg, self.tpb](self.system.d_x,
                                                 self.system.d_box,
                                                 self.d_ibox,
                                                 self.d_cell_list,
                                                 self.d_cell_list_index,
                                                 self.d_cell_counts,
                                                 self.d_cells,
                                                 self.d_cell_max)
                self.d_cell_max.copy_to_host(self.p_cell_max)
                cuda.synchronize()
                if self.p_cell_max[0] > self.cell_guess:
                    self.cell_guess = self.p_cell_max[0]
                    self.cell_guess = self.cell_guess + 8 - (self.cell_guess & 7)
                    self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess, self.system.n_dim),
                                                         dtype=self.system.dtype)
                    self.d_cell_list_index = cuda.device_array((self.ncell, self.cell_guess), dtype=np.int32)
                else:
                    break
