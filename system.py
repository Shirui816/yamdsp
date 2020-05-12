from numba import cuda
import numpy as np

class system:
    def __init__(self, x, box, typid, bond, gpu=0):
        self.x = x
        self.box = box
        self.N = x.shape[0]
        self.n_dim = x.shape[1]
        self.gpu = gpu
        self.typid = typid
        self.types = np.asarray(list(set(typid)), dtype=np.int32)
        self.bond = bond
        self.bonds = np.asarray(list(set(bond.T[0])), dtype=np.int32)
        with cuda.gpus[gpu]:
            self.d_x = cuda.to_device(x)
            self.d_box = cuda.to_device(box)
            self.d_typid = cuda.to_device(typid)
            self.d_types = cuda.to_device(self.types)
            self.d_bond = cuda.to_device(self.bond)
            self.d_bonds = cuda.to_device(self.bonds)


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
