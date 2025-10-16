from mpi4py import MPI
import timeit
import math
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = MPI.Wtime()

def y(x):
    return (4/(1+x**2))

def midpoint(a, b, n):
    h = (b - a) / n
    total = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        total += y(x_mid)
    return h * total

a = 0
b = 1
n = 10**9
h = (b - a)/n

local_n = n // size
local_a = a + rank * local_n * h
local_b = local_a + local_n * h

local_integral = midpoint(local_a, local_b, local_n)
print("hasil local_integral = ", local_integral, "pada proses", rank)
my_integral=0
if rank == 0:
    my_integral = local_integral
    for i in range(1,size):
        my_integral_2 = comm.recv(source=MPI.ANY_SOURCE)
        my_integral = my_integral + my_integral_2
    print("hasil numerik = ", my_integral)
    print("waktu = ", MPI.Wtime()-start, "detik")
else:
    comm.send(local_integral,dest=0)