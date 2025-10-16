from mpi4py import MPI
import timeit
import math
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = MPI.Wtime()

def y(x):
    return (4/(1+x**2))

def simpson(a, b, n): 
    h = (b - a) / n
    s = y(a) + y(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            s += 2 * y(x)  # untuk indeks genap
        else:
            s += 4 * y(x)  # untuk indeks ganjil
    
    return (h / 3) * s

a = 0
b = 1
n = 10**9
h = (b - a)/n

local_n = n // size
local_a = a + rank * local_n * h
local_b = local_a + local_n * h

local_integral = simpson(local_a, local_b, local_n)

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