from numba import cuda

@cuda.jit("void(int32[:], int32)")
def cu_set_to_int(arr, val):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    arr[i] = val


@cuda.jit("void(float64[:], float64)")
def cu_set_to_float(arr, val):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    arr[i] = val