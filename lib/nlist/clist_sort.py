from math import floor

import cupy
import numpy as np
from numba import cuda
from numba import int32

from lib._helpers import Ctx
from lib.utils import cu_unravel_index_f, cu_ravel_index_f_pbc
from . import cu_set_to_int


@cuda.jit("int32(float64[:], float64[:], int32[:])", device=True)
def cu_cell_index(x, box, ibox):
    ret = floor((x[0] / box[0] + 0.5) * ibox[0])
    n_cell = ibox[0]
    for i in range(1, x.shape[0]):
        ret = ret + floor((x[i] / box[i] + 0.5) * ibox[i]) * n_cell
        n_cell = n_cell * ibox[i]
    return ret


def gen_cell_map(ndim):
    @cuda.jit("void(int32[:], int32[:], int32[:, :])")
    def cu_cell_map(ibox, dim, ret):
        cell_i = cuda.grid(1)
        if cell_i >= ret.shape[0]:
            return
        cell_vec_i = cuda.local.array(ndim, int32)
        cell_vec_j = cuda.local.array(ndim, int32)
        cu_unravel_index_f(cell_i, ibox, cell_vec_i)
        for j in range(ret.shape[1]):
            cu_unravel_index_f(j, dim, cell_vec_j)
            for k in range(ndim):
                cell_vec_j[k] = cell_vec_i[k] + cell_vec_j[k] - 1
            cell_j = cu_ravel_index_f_pbc(cell_vec_j, ibox)
            ret[cell_i, j] = cell_j

    return cu_cell_map


@cuda.jit("void(float64[:, :], float64[:], int32[:], int32[:], int32[:])")
def cu_cell_list(pos, box, ibox, cells, cell_counts):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        pi = pos[i]
        ic = cu_cell_index(pi, box, ibox)
        cells[i] = ic
        cuda.atomic.add(cell_counts, ic + 1, 1)


class clist:
    def __init__(self, r_cut, r_buff=0.5):
        system = Ctx.get_active()
        if system is None:
            raise ValueError("No active system!")
        self.system = system
        self.r_cut = r_cut
        self.r_buff = r_buff
        self.ibox = np.asarray(np.floor(system.box / (r_cut + r_buff)), dtype=np.int32)
        self.n_cell = int(np.multiply.reduce(self.ibox))
        self.cell_adj = np.ones(system.n_dim, dtype=np.int32) * 3
        self.gpu = system.gpu
        self.tpb = 64
        self.bpg_part = int(system.N / self.tpb + 1)
        self.bpg_cell = int((self.n_cell + 1) // self.tpb + 1)
        cu_cell_map = gen_cell_map(system.n_dim)
        self.d_cell_list = None
        with cuda.gpus[self.gpu]:
            self.d_cells = cupy.empty((system.N,), dtype=np.int32)
            self.d_cell_map = cuda.device_array((self.n_cell, 3 ** system.n_dim), dtype=np.int32)
            self.d_ibox = cuda.to_device(self.ibox)
            self.d_cell_adj = cuda.to_device(self.cell_adj)
            cu_cell_map[self.bpg_cell, self.tpb](self.d_ibox, self.d_cell_adj, self.d_cell_map)
            self.d_cell_counts = cupy.zeros(self.n_cell + 1, dtype=np.int32)
            # cu_cell_list[self.bpg_part, self.tpb](system.d_x, system.d_box, self.d_ibox,
            #                                      self.d_cells, self.d_cell_counts)
            # cupy.cumsum(self.d_cell_counts, out=self.d_cell_counts)
            # self.d_cell_list = cupy.argsort(self.d_cells)
        self.update()

    def update(self):
        with cuda.gpus[self.gpu]:
            cu_set_to_int[self.bpg_cell, self.tpb](self.d_cell_counts, 0)
            cu_cell_list[self.bpg_part, self.tpb](self.system.d_x, self.system.d_box, self.d_ibox,
                                                  self.d_cells, self.d_cell_counts)
            cupy.cumsum(self.d_cell_counts, out=self.d_cell_counts)
            self.d_cell_list = cupy.argsort(self.d_cells)
