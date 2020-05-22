import numpy as np

from lib.system import system

x = np.loadtxt('data/pos.txt').astype(np.float32)
box = np.array([50, 50, 50.]).astype(np.float32)
typ = np.ones(x.shape[0])
s = system(x, box, typ)

from lib.nlist.nlist_sort import nlist as nlist_sort
from lib.nlist.nlist import nlist as nlist

nlist_s = nlist_sort(3.0, 0.5)
ret_s = nlist_s.show()
nl_s, nc_s = ret_s[-2], ret_s[-1]
print(nl_s[0, :nc_s[0]], nc_s[0])

nlist_a = nlist(3.0, 0.5)
ret_a = nlist_a.show()
nl_a, nc_a = ret_a[-2], ret_a[-1]
print(nl_a[0, :nc_a[0]], nc_a[0])

nl_a.sort(axis=-1)
nl_s.sort(axis=-1)
print(np.allclose(nl_a, nl_s))

import time

nloop = 1000
s = time.time()
for i in range(nloop):
    nlist_s.update(forced=True)
    # nlist_a.update(forced=True)
print(nloop/(time.time() - s), "sort")
s = time.time()
for i in range(nloop):
    nlist_a.update(forced=True)
    # nlist_a.update(forced=True)
print(nloop/(time.time() - s), "non-sort")
