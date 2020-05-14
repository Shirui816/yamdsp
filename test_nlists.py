from lib.system import system
import numpy as np

x = np.loadtxt('data/pos.txt')
box = np.array([50,50,50.])
typ = np.ones(x.shape[0])
s = system(x, box, typ)

from lib.nlist.clist_sort import clist as clist_sort
#from yamdsp.nlist.clist import clist as clist

#clist_s = clist_sort(3.0, 0.5)
#print(clist_s.d_cell_list, clist_s.d_cell_list.shape)
#clist_a = clist(3.0, 0.5)

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

import time
s = time.time()
for i in range(1000):
    nlist_s.update(forced=True)
    #nlist_a.update(forced=True)
print(time.time()-s, "sort")
for i in range(1000):
    nlist_a.update(forced=True)
    #nlist_a.update(forced=True)
print(time.time()-s, "non-sort")
