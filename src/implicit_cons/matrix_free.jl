using SparseArrays, IterativeSolvers, BenchmarkTools, LinearMaps, Test


"""
construct_A_RHS(dx, dy, Nx, Ny, f, un)

Construct the matrix A and the right-hand-side b.
This version is slow because of the low efficient filling of sparse matrix.
"""
function construct_A_RHS(dx, dy, Nx, Ny, f, un)
   N = (Nx-1)*(Ny-1) # total # of unknowns (excluding boundaries)
   A = spzeros(N,N)  # matrix representing the Laplace operator

   # intermediate variables
   diag = -2/dx^2 - 2/dy^2
   invdx2, invdy2 = 1/dx^2, 1/dy^2

   b = vec(f[2:end-1,2:end-1])

   # fill in interior contributions & apply B.C. to RHS
   for iy = 2:Ny, ix = 2:Nx
      i = (iy-2)*(Nx-1) + ix - 1
      iL, iR, iD, iU = i-1, i+1, i-(Nx-1), i+(Nx-1)
      A[i,i] = diag
      if ix > 2
         A[i,iL] = invdx2
      else
         b[i] -= un[1,iy] * invdx2
      end
      if ix < Nx
         A[i,iR] = invdx2
      else
         b[i] -= un[end,iy] * invdx2
      end
      if iy > 2
         A[i,iD] = invdy2
      else
         b[i] -= un[ix,1] * invdy2
      end
      if iy < Ny
         A[i,iU] = invdy2
      else
         b[i] -= un[ix,end] * invdy2
      end
   end

   return A, b
end

"""
construct_RHS(dx, dy, Nx, Ny, f, un)

Construct the right-hand-side b.
"""
function construct_RHS(dx, dy, Nx, Ny, f, un)

   # intermediate variables
   invdx2, invdy2 = 1/dx^2, 1/dy^2

   b = vec(f[2:end-1,2:end-1])

   # apply B.C. to RHS
   for iy = 2:Ny, ix = 2:Nx
      i = (iy-2)*(Nx-1) + ix - 1
      if ix ≤ 2
         b[i] -= un[1,iy] * invdx2
      end
      if ix ≥ Nx
         b[i] -= un[end,iy] * invdx2
      end
      if iy ≤ 2
         b[i] -= un[ix,1] * invdy2
      end
      if iy ≥ Ny
         b[i] -= un[ix,end] * invdy2
      end
   end

   return b
end


nx = 4
ny = 4

x_l = 0.0
x_r = 1.0
y_b = 0.0
y_t = 1.0

dx = (x_r - x_l)/nx
dy = (y_t - y_b)/ny

# allocate array for x and y position of grids, exact solution and source term
x  = Array{Float64}(undef, nx+1)
y  = Array{Float64}(undef, ny+1)
ue = Array{Float64}(undef, nx+1, ny+1)
f  = Array{Float64}(undef, nx+1, ny+1)
un = Array{Float64}(undef, nx+1, ny+1)

for i = 1:nx+1
   x[i] = x_l + dx*(i-1)
end
for i = 1:ny+1
   y[i] = y_b + dy*(i-1)
end

for j = 1:ny+1, i = 1:nx+1
   ue[i,j] = (x[i]^2 - 1.0)*(y[j]^2 - 1.0)

   f[i,j] = -2.0*(2.0 - x[i]^2 - y[j]^2)

   un[i,j] = 0.0
end

# Dirichlet B.C.
un[:,1] = ue[:,1]
un[:,ny+1] = ue[:,ny+1]

un[1,:] = ue[1,:]
un[nx+1,:] = ue[nx+1,:]

r = zeros(Float64, nx+1, ny+1)
init_rms = 0.0
rms = 0.0

# sparse matrix form
A1, b = construct_A_RHS(dx, dy, nx, ny, f, un)


# matrix-free form
diag = -2/dx^2 - 2/dy^2
invdx2, invdy2 = 1/dx^2, 1/dy^2

constructA = (dx, dy, Nx, Ny) -> LinearMap((Nx-1)*(Ny-1); issymmetric=true, ismutating=true) do C,B
   m = Nx - 1

   for i = 1:length(B)
      C[i] = diag * B[i]
   end
   for i = 1:length(B)-1
      if i % m != 0
         C[i] += invdx2 * B[i+1]
         C[i+1] += invdx2 * B[i]
      end
   end
   for i = 1:(length(B)-m)
      C[i] += invdy2 * B[i+m]
      C[i+m] += invdy2 * B[i]
   end
end

b = construct_RHS(dx, dy, nx, ny, f, un)
A2 = constructA(dx, dy, nx, ny)


## Test
@btime U1 = cg(A1, b);
@btime U2 = cg(A2, b); # too much memory usage!!!

U1 = cg(A1, b)
U2 = cg(A2, b)
@test U1 ≈ U2
