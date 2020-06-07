from math import floor

import numpy as np
from numba import cuda
from numba import int32, float32, float64, void

from . import cu_set_to_int
from .._helpers import Ctx
from ..utils import cu_unravel_index_f, cu_ravel_index_f_pbc


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
        self.cu_cell_map, self.cu_cell_list = self._gen_func()
        self.p_cell_max = cuda.pinned_array((1,), dtype=np.int32)
        self.p_out_of_box = cuda.pinned_array((1,), dtype=self.system.dtype)
        with cuda.gpus[self.gpu]:
            self.d_cells = cuda.device_array(self.system.d_x.shape[0], dtype=np.int32)
            self.d_cell_map = cuda.device_array((self.n_cell, 3 ** system.n_dim), dtype=np.int32)
            self.d_ibox = cuda.to_device(self.ibox)
            self.d_cell_adj = cuda.to_device(self.cell_adj)
            self.cu_cell_map[self.bpg_cell, self.tpb](self.d_ibox, self.d_cell_adj, self.d_cell_map)
            self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess),
                                                 dtype=np.int32)
            self.d_cell_counts = cuda.device_array(self.n_cell, dtype=np.int32)
            self.d_cell_max = cuda.device_array(1, dtype=np.int32)
            self.d_out_of_box = cuda.device_array(1, dtype=self.system.dtype)
        self.update()

    def update(self):
        with cuda.gpus[self.gpu]:
            while True:
                self.p_out_of_box[0] = -1
                cu_set_to_int[self.bpg_cell, self.tpb](self.d_cell_counts, 0)
                self.cu_cell_list[self.bpg, self.tpb](self.system.d_x,
                                                      self.system.d_box,
                                                      self.d_ibox,
                                                      self.d_cell_list,
                                                      self.d_cell_counts,
                                                      self.d_cells,
                                                      self.d_cell_max,
                                                      self.d_out_of_box)
                self.d_cell_max.copy_to_host(self.p_cell_max)
                self.d_out_of_box.copy_to_host(self.p_out_of_box)
                cuda.synchronize()
                if self.p_out_of_box[0] != -1:
                    err_coor = ''.join(["{0: .4f} ".format(_) for _ in self.p_out_of_box[1:]])
                    raise ValueError("Error, particle %d %s is out of box!" % (int(self.p_out_of_box[0]), err_coor))
                if self.p_cell_max[0] > self.cell_guess:
                    self.cell_guess = self.p_cell_max[0]
                    self.cell_guess = self.cell_guess + 8 - (self.cell_guess & 7)
                    self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess),
                                                         dtype=np.int32)
                else:
                    break

    def _gen_func(self):
        n_dim = self.system.n_dim
        nb_float = float64
        if self.system.dtype == np.dtype(np.float32):
            nb_float = float32

        @cuda.jit(int32(nb_float[:], nb_float[:], int32[:]), device=True)
        def cu_cell_index(x, box, ibox):
            ret = 0
            n_cell = 1
            for i in range(0, x.shape[0]):
                tmp = x[i] / box[i]
                if tmp < -0.5 or tmp > 0.5:
                    return -1
                ret = ret + floor((tmp + 0.5) * ibox[i]) * n_cell
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

        @cuda.jit(void(nb_float[:, :], nb_float[:], int32[:], int32[:, :], int32[:], int32[:], int32[:], nb_float[:]))
        def cu_cell_list(x, box, ibox, cell_list, cell_counts, cells, cell_max, out_of_box):
            pi = cuda.grid(1)
            if pi >= x.shape[0]:
                return
            # xi = cuda.local.array(ndim, dtype=float64)
            # for k in range(ndim):
            # xi[k] = x[pi, k]
            xi = x[pi]
            ic = cu_cell_index(xi, box, ibox)
            if ic == -1:
                out_of_box[0] = nb_float(pi)
                for d in range(n_dim):
                    out_of_box[d + 1] = x[pi, d]
            cells[pi] = ic
            index = cuda.atomic.add(cell_counts, ic, 1)
            if index < cell_list.shape[0]:
                cell_list[ic, index] = pi
            else:
                cuda.atomic.max(cell_max, 0, index + 1)

        return cu_cell_map, cu_cell_list
