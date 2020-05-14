import gc

import numpy as np
from numba import cuda

from lib._helpers import Ctx


class system:
    def __init__(self, x, box, typ, bond=None, diameter=None, gpu=0, num=None):
        self.x = x
        self.box = box
        self.N = x.shape[0]
        self.n_dim = x.shape[1]
        self.gpu = gpu
        typ = np.asarray(typ)
        self.types = list(set(typ))
        self.typid = np.zeros(self.N, dtype=np.int32)
        for i, t in enumerate(self.types):
            self.typid[typ == t] = i
        if bond:
            self.bond = bond
            self.bonds = np.asarray(list(set(bond.T[0])), dtype=np.int32)
        self.diameter = diameter if diameter is not None else np.ones(self.N, dtype=np.float64)
        with cuda.gpus[gpu]:
            self.d_x = cuda.to_device(x)
            self.d_box = cuda.to_device(box)
            self.d_typid = cuda.to_device(self.typid)
            # self.d_types = cuda.to_device(self.types)
            if bond:
                self.d_bond = cuda.to_device(bond)
                self.d_bonds = cuda.to_device(self.bonds)
            self.d_diameter = cuda.to_device(diameter)
            self.d_force = cuda.device_array((self.N, self.n_dim), dtype=np.float64)
        if num is None:
            system.num = Ctx.get_num_systems() + 1
        else:
            if Ctx.has_num(num):
                raise ValueError("Number %d has already been registered!"
                                 "Please assign new number of system." % num)
        Ctx.set_active(self)

    def bond_table(self):
        pass

    def molecules(self):
        pass

    def snapshot(self):
        self.x = self.d_x.copy_to_host()
        self.box = self.d_box.copy_to_host()
        # whatever has been changed
        cuda.synchronize()
        return self

    def destroy(self):
        # delete variables in self.
        gc.collect(1)
