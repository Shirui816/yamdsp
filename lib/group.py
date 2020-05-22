from functools import reduce

import numpy as np

from ._helpers import Ctx


class group:
    def __init__(self, name=None, typ=None, indices=None):
        self.name = name if name is not None else typ
        self.typ = typ
        if (name is None and typ is None) or (typ is not None and indices is not None):
            raise ValueError("Are you serious?")
        system = Ctx.get_active()
        if not system:
            raise ValueError("Initialize system first!")
        if typ is None:
            self.indices = indices
        else:
            self.indices = np.arange(system.N)[system.typid == system.types.index(typ)]

    def union(self, name=None, *args):
        if name is None:
            name = self.name + reduce(str.__add__, [_.name for _ in args])
        indices = reduce(np.union1d, (_.indices for _ in args))
        return group(name=name, typ=None, indices=indices)
