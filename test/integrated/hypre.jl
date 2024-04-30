using MPI, HYPRE, SparseArrays
MPI.Init()

n = 100
A = sprand(n, n, 0.7)
b = rand(n)
# comm = MPI.COMM_WORLD

# Preconditioner
precond = HYPRE.BoomerAMG(; RelaxType=6, CoarsenType=6)

# Solver
solver = HYPRE.GMRES(; MaxIter=1000, Tol=1e-9, Precond=precond)

x = HYPRE.solve(solver, A, b)
