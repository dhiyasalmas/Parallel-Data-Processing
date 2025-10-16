from mpi4py import MPI
import numpy as np
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = MPI.Wtime()

def y(x):
    return 4 / (1 + x**2)

def trapezoid(a, b, n):
    h = (b - a) / n
    s = y(a) + y(b)
    for i in range(1, n):
        s += 2 * y(a + i * h)
    return (h / 2) * s

def romberg(a, b, max_k=5):
    R = np.zeros((max_k, max_k))
    for k in range(max_k):
        n = 2**k
        R[k, 0] = trapezoid(a, b, n)
        for j in range(1, k + 1):
            R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
    return R[max_k-1, max_k-1]

a = 0
b = 1
n = 10**6   
h = (b - a) / n

local_n = n // size
local_a = a + rank * local_n * h
local_b = local_a + local_n * h

local_integral = romberg(local_a, local_b)

my_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)

if rank == 0:
    print("Hasil numerik =", my_integral)
    print("Waktu =", MPI.Wtime() - start, "detik")
