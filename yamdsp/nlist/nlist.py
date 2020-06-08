import numpy as np
from numba import cuda, void, int32, float32, float64

from . import cu_set_to_int
from .clist import clist
from .._helpers import Ctx


@cuda.jit(void(int32[:], int32[:]))
def cu_max_int(arr, arr_max):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    cuda.atomic.max(arr_max, 0, arr[i])


class nlist(object):
    def __init__(self, r_cut, r_buff=0.5, cell_guess=50, n_guess=150):
        system = Ctx.get_active()
        if system is None:
            raise ValueError("No active system!")
        self.system = system
        self.cell_guess = cell_guess
        self.n_guess = n_guess
        self.r_cut2 = r_cut ** 2
        self.r_buff2 = (r_buff / 2) ** 2
        self.gpu = system.gpu
        self.tpb = 64
        self.bpg = int(self.system.N // self.tpb + 1)
        # self.situ_zero = np.zeros(1, dtype=np.int32)
        self.update_counts = 0
        self.cu_nlist, self.cu_check_build = self._gen_func()
        with cuda.gpus[self.gpu]:
            self.p_n_max = cuda.pinned_array((1,), dtype=np.int32)
            self.p_situation = cuda.pinned_array((1,), dtype=np.int32)
            self.d_last_x = cuda.device_array_like(self.system.d_x)
            self.d_n_max = cuda.device_array(1, dtype=np.int32)
            self.d_nl = cuda.device_array((self.system.N, self.n_guess), dtype=np.int32)
            self.d_nc = cuda.device_array((self.system.N,), dtype=np.int32)
            self.d_situation = cuda.device_array(1, dtype=np.int32)
        self.clist = clist(r_cut, r_buff, cell_guess=self.cell_guess)
        self.neighbour_list()

    def neighbour_list(self):
        with cuda.gpus[self.gpu]:
            while True:
                cu_set_to_int[self.bpg, self.tpb](self.d_nc, 0)
                # reset situation while build nlist
                self.cu_nlist[self.bpg, self.tpb](self.system.d_x,
                                                  self.d_last_x,
                                                  self.system.d_box,
                                                  self.r_cut2,
                                                  self.clist.d_cell_map,
                                                  self.clist.d_cell_list,
                                                  self.clist.d_cell_counts,
                                                  self.clist.d_cells,
                                                  self.d_nl,
                                                  self.d_nc,
                                                  self.d_n_max,
                                                  self.d_situation)
                self.d_n_max.copy_to_host(self.p_n_max)
                cuda.synchronize()
                # n_max = np.array([120])
                if self.p_n_max[0] > self.n_guess:
                    self.n_guess = self.p_n_max[0]
                    self.n_guess = self.n_guess + 8 - (self.n_guess & 7)
                    self.d_nl = cuda.device_array((self.system.N, self.n_guess), dtype=np.int32)
                else:
                    break

    def check_update(self):
        with cuda.gpus[self.gpu]:
            self.cu_check_build[self.bpg, self.tpb](self.system.d_x, self.system.d_box, self.d_last_x, self.r_buff2,
                                                    self.d_situation)
        self.d_situation.copy_to_host(self.p_situation)
        cuda.synchronize()
        return self.p_situation

    def update(self, forced=False):
        if not forced:
            s = self.check_update()
        else:
            s = [1]
        if s[0] == 1:
            self.clist.update()
            self.neighbour_list()
            self.update_counts += 1

    def show(self):
        cell_list = self.clist.d_cell_list.copy_to_host()
        cell_map = self.clist.d_cell_map.copy_to_host()
        cell_counts = self.clist.d_cell_counts.copy_to_host()
        nl = self.d_nl.copy_to_host()
        nc = self.d_nc.copy_to_host()
        cuda.synchronize()
        return cell_list, cell_counts, cell_map, nl, nc

    def _gen_func(self):
        from math import floor
        nb_float = float64
        if self.system.dtype == np.dtype(np.float32):
            nb_float = float32

        @cuda.jit(nb_float(nb_float[:], nb_float[:], nb_float[:]), device=True)
        def cu_pbc_dist2(a, b, box):
            ret = 0
            for i in range(a.shape[0]):
                d = a[i] - b[i]
                d -= box[i] * floor(d / box[i] + 0.5)
                ret += d ** 2
            return ret

        @cuda.jit(
            void(nb_float[:, :], nb_float[:, :], nb_float[:], nb_float, int32[:, :], int32[:, :], int32[:], int32[:],
                 int32[:, :], int32[:], int32[:], int32[:]))
        def cu_nlist(x, last_x, box, r_cut2, cell_map, cell_list, cell_count, cells, nl, nc, n_max,
                     situation):
            pi = cuda.grid(1)
            if pi >= x.shape[0]:
                return
            # xi = cuda.local.array(ndim, dtype=float64)
            # xj = cuda.local.array(ndim, dtype=float64)
            # for l in range(ndim):
            #    xi[l] = x[pi, l]
            ic = cells[pi]
            n_needed = 0
            nn = 0
            xi = x[pi]
            for j in range(cell_map.shape[1]):
                jc = cell_map[ic, j]
                for k in range(cell_count[jc]):
                    pj = cell_list[jc, k]
                    if pj == pi:
                        continue
                    # for m in range(ndim):
                    # xj[m] = x[pj, m]
                    r2 = cu_pbc_dist2(xi, x[pj], box)
                    if r2 < r_cut2:
                        if nn < nl.shape[1]:
                            nl[pi, nn] = pj
                        else:
                            n_needed = nn + 1
                        nn += 1
            nc[pi] = nn
            for k in range(x.shape[1]):
                last_x[pi, k] = x[pi, k]
            if nn > 0:
                cuda.atomic.max(n_max, 0, n_needed)
            if pi == 0:  # reset situation only once while function is called
                situation[0] = 0

        @cuda.jit(void(nb_float[:, :], nb_float[:], nb_float[:, :], nb_float, int32[:]))
        def cu_check_build(x, box, last_x, r_buff2, situation):
            # rebuild lists if there exist particles move larger than buffer
            i = cuda.grid(1)
            if i < x.shape[0]:
                dr2 = cu_pbc_dist2(x[i], last_x[i], box)
                if dr2 > r_buff2:
                    situation[0] = 1

        return cu_nlist, cu_check_build
