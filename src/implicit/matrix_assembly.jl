using SparseArrays
using StaticArrays
using AlgebraicMultigrid
using Krylov
using LinearSolve
using ILUZero
using LinearAlgebra
using OffsetArrays

function diffusion_op_2nd_order_2d_zerogradient_bc(
  edge_metrics, a_edge::SVector{4,T}, loc
) where {T}

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section
  stencil = MMatrix{3,3,T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  # Diff(I) term ∂/∂ξ [... ∂/∂ξ]
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  # Diff(IV) term ∂/∂η [... ∂/∂η]
  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  # i boundaries
  if loc === :ilo || loc === :ilojlo || loc === :ilojhi
    fᵢ₋½ = zero(T)
  elseif loc === :ihi || loc === :ihijlo || loc === :ihijhi
    fᵢ₊½ = zero(T)
  end

  stencil[1, 2] += fᵢ₋½           # u[i-1, j]
  stencil[2, 2] += -(fᵢ₊½ + fᵢ₋½) # u[i  , j]
  stencil[3, 2] += fᵢ₊½           # u[i+1, j]

  # j boundaries
  if loc === :jlo || loc === :ilojlo || loc === :ihijlo
    fⱼ₋½ = zero(T)
  elseif loc === :jhi || loc === :ilojhi || loc === :ihijhi
    fⱼ₋½ = zero(T)
  end

  stencil[2, 1] += fⱼ₋½           # u[i, j-1]
  stencil[2, 2] += -(fⱼ₊½ + fⱼ₋½) # u[i, j  ]
  stencil[2, 3] += fⱼ₊½           # u[i, j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # TODO: pure rectangular grids can skip this section

  # Diff(II) term  ∂/∂ξ [... ∂/∂η] -> only apply this on i boundaries,
  # since ∂/∂η = 0 at all the j boundaries
  if (loc === :ilo || loc === :ihi)
    gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
    gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
    # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
    #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

    stencil[1, 1] += gᵢ₋½           # u[i-1, j-1]
    stencil[2, 1] += (-gᵢ₊½ + gᵢ₋½) # u[i  , j-1]
    stencil[3, 1] += -gᵢ₊½          # u[i+1, j-1]
    stencil[1, 3] += -gᵢ₋½          # u[i-1, j+1]
    stencil[2, 3] += (gᵢ₊½ - gᵢ₋½)  # u[i  , j+1]
    stencil[3, 3] += gᵢ₊½           # u[i+1, j+1]
  end

  # Diff(III) term  ∂/∂η [... ∂/∂ξ] -> only apply this on j boundaries,
  # since ∂/∂ξ = 0 at all the i boundaries
  if (loc === :jlo || loc === :jhi)
    gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
    gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
    # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
    #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

    stencil[1, 1] += gⱼ₋½           # u[i-1, j-1]
    stencil[3, 1] += -gⱼ₋½          # u[i+1, j-1]
    stencil[1, 2] += (-gⱼ₊½ + gⱼ₋½) # u[i-1, j  ]
    stencil[3, 2] += (gⱼ₊½ - gⱼ₋½)  # u[i+1, j  ]
    stencil[1, 3] += -gⱼ₊½          # u[i-1, j+1]
    stencil[3, 3] += gⱼ₊½           # u[i+1, j+1]
  end

  # return the offset version to make indexing easier
  # (no speed penalty using an offset array here)
  return OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)
end

function inner_diffusion_operator_2d(edge_metrics, a_edge::SVector{4,T}) where {T}

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]

  # Don't use an offset matrix here b/c benchmarking showed it was
  # ~3x slower. It makes the indexing here more of a pain,
  # but 3x slower is a big deal for this performance critical section
  stencil = MMatrix{3,3,T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

  # Here the Δξ and Δη terms are 1
  a_ᵢ₊½, a_ᵢ₋½, a_ⱼ₊½, a_ⱼ₋½ = a_edge

  # edge_metrics = metrics_2d(mesh, i, j)
  Jᵢ₊½ = edge_metrics.Jᵢ₊½
  Jᵢ₋½ = edge_metrics.Jᵢ₋½
  Jⱼ₊½ = edge_metrics.Jⱼ₊½
  Jⱼ₋½ = edge_metrics.Jⱼ₋½
  Jξx_ᵢ₊½ = edge_metrics.Jξx_ᵢ₊½
  Jξy_ᵢ₊½ = edge_metrics.Jξy_ᵢ₊½
  Jηx_ᵢ₊½ = edge_metrics.Jηx_ᵢ₊½
  Jηy_ᵢ₊½ = edge_metrics.Jηy_ᵢ₊½
  Jξx_ᵢ₋½ = edge_metrics.Jξx_ᵢ₋½
  Jξy_ᵢ₋½ = edge_metrics.Jξy_ᵢ₋½
  Jηx_ᵢ₋½ = edge_metrics.Jηx_ᵢ₋½
  Jηy_ᵢ₋½ = edge_metrics.Jηy_ᵢ₋½
  Jξx_ⱼ₊½ = edge_metrics.Jξx_ⱼ₊½
  Jξy_ⱼ₊½ = edge_metrics.Jξy_ⱼ₊½
  Jηx_ⱼ₊½ = edge_metrics.Jηx_ⱼ₊½
  Jηy_ⱼ₊½ = edge_metrics.Jηy_ⱼ₊½
  Jξx_ⱼ₋½ = edge_metrics.Jξx_ⱼ₋½
  Jξy_ⱼ₋½ = edge_metrics.Jξy_ⱼ₋½
  Jηx_ⱼ₋½ = edge_metrics.Jηx_ⱼ₋½
  Jηy_ⱼ₋½ = edge_metrics.Jηy_ⱼ₋½

  #------------------------------------------------------------------------------
  # Equations 3.43 and 3.44
  #------------------------------------------------------------------------------
  fᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½^2 + Jξy_ᵢ₊½^2) / Jᵢ₊½
  fᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½^2 + Jξy_ᵢ₋½^2) / Jᵢ₋½
  # i terms = fᵢ₋½ * u[i-1,j] - (fᵢ₊½ + fᵢ₋½) * u[i,j] + fᵢ₊½ * u[i+1,j]

  stencil[1, 2] += fᵢ₋½           # u[i-1, j]
  stencil[2, 2] += -(fᵢ₊½ + fᵢ₋½) # u[i  , j]
  stencil[3, 2] += fᵢ₊½           # u[i+1, j]

  fⱼ₊½ = a_ⱼ₊½ * (Jηx_ⱼ₊½^2 + Jηy_ⱼ₊½^2) / Jⱼ₊½
  fⱼ₋½ = a_ⱼ₋½ * (Jηx_ⱼ₋½^2 + Jηy_ⱼ₋½^2) / Jⱼ₋½
  # j terms = fⱼ₋½ * u[i,j-1] - (fⱼ₊½ + fⱼ₋½) * u[i,j] + fⱼ₊½ * u[i,j+1]

  stencil[2, 1] += fⱼ₋½           # u[i, j-1]
  stencil[2, 2] += -(fⱼ₊½ + fⱼ₋½) # u[i, j  ]
  stencil[2, 3] += fⱼ₊½           # u[i, j+1]

  #------------------------------------------------------------------------------
  # cross terms (Equations 3.45 and 3.46)
  #------------------------------------------------------------------------------
  # TODO: pure rectangular grids can skip this section
  gᵢ₊½ = a_ᵢ₊½ * (Jξx_ᵢ₊½ * Jηx_ᵢ₊½ + Jξy_ᵢ₊½ * Jηy_ᵢ₊½) / (4Jᵢ₊½)
  gᵢ₋½ = a_ᵢ₋½ * (Jξx_ᵢ₋½ * Jηx_ᵢ₋½ + Jξy_ᵢ₋½ * Jηy_ᵢ₋½) / (4Jᵢ₋½)
  # i terms = gᵢ₊½ * (u[i, j+1] − u[i, j-1] + u[i+1, j+1] − u[i+1, j-1])
  #          -gᵢ₋½ * (u[i, j+1] − u[i, j-1] + u[j−1, j+1] − u[j−1, j-1])

  stencil[1, 1] += gᵢ₋½           # u[i-1, j-1]
  stencil[2, 1] += (-gᵢ₊½ + gᵢ₋½) # u[i  , j-1]
  stencil[3, 1] += -gᵢ₊½          # u[i+1, j-1]
  stencil[1, 3] += -gᵢ₋½          # u[i-1, j+1]
  stencil[2, 3] += (gᵢ₊½ - gᵢ₋½)  # u[i  , j+1]
  stencil[3, 3] += gᵢ₊½           # u[i+1, j+1]

  gⱼ₊½ = a_ⱼ₊½ * (Jξx_ⱼ₊½ * Jηx_ⱼ₊½ + Jξy_ⱼ₊½ * Jηy_ⱼ₊½) / (4Jⱼ₊½)
  gⱼ₋½ = a_ⱼ₋½ * (Jξx_ⱼ₋½ * Jηx_ⱼ₋½ + Jξy_ⱼ₋½ * Jηy_ⱼ₋½) / (4Jⱼ₋½)
  # j terms = gⱼ₊½ * (u[i+1, j] − u[i-1, j] + u[i+1, j+1] − u[i-1, j+1])
  #          -gⱼ₋½ * (u[i+1, j] − u[i-1, j] + u[i+1, j-1] − u[i-1, j-1])

  stencil[1, 1] += gⱼ₋½           # u[i-1, j-1]
  stencil[3, 1] += -gⱼ₋½          # u[i+1, j-1]
  stencil[1, 2] += (-gⱼ₊½ + gⱼ₋½) # u[i-1, j  ]
  stencil[3, 2] += (gⱼ₊½ - gⱼ₋½)  # u[i+1, j  ]
  stencil[1, 3] += -gⱼ₊½          # u[i-1, j+1]
  stencil[3, 3] += gⱼ₊½           # u[i+1, j+1]

  # return the offset version to make indexing easier
  # (no speed penalty using an offset array here)
  return OffsetMatrix(SMatrix{3,3}(stencil), -1:1, -1:1)
end

function assemble_matrix!(A) end

ni, nj = 6, 6

len = ni * nj
A = spdiagm(
  -ni - 1 => ones(len - ni - 1), # (i-1, j-1)
  -ni => ones(len - ni),         # (i  , j-1)
  -ni + 1 => ones(len - ni + 1), # (i+1, j-1)
  -1 => ones(len - 1),           # (i-1, j  )
  0 => 4ones(len),               # (i  , j  )
  1 => ones(len - 1),            # (i+1, j  )
  ni - 1 => ones(len - ni + 1),  # (i-1, j+1)
  ni => ones(len - ni),          # (i  , j+1)
  ni + 1 => ones(len - ni - 1),  # (i+1, j+1)
)

for row in eachrow(A)
  @show row
end

nh = 1
CI = CartesianIndices((1:ni, 1:nj))
LI = LinearIndices(CI)
innerLI = LI[(nh + 1):(end - nh), (nh + 1):(end - nh)]

T = Float64
_stencil = MMatrix{3,3,T}(11, 21, 31, 12, 22, 32, 13, 23, 33)

stencil = OffsetMatrix(SMatrix{3,3}(_stencil), -1:1, -1:1)

for idx in innerLI
  i, j = CI[idx].I
  #! format: off
  A[idx, idx - ni - 1] = stencil[-1, -1] # (i-1, j-1)
  A[idx, idx - ni]     = stencil[+0, -1] # (i  , j-1)
  A[idx, idx - ni + 1] = stencil[+1, -1] # (i+1, j-1)
  A[idx, idx - 1]      = stencil[-1, +0] # (i-1, j  )
  A[idx, idx]          = stencil[+0, +0] # (i  , j  )
  A[idx, idx + 1]      = stencil[+1, +0] # (i+1, j  )
  A[idx, idx + ni + 1] = stencil[-1, +1] # (i-1, j+1)
  A[idx, idx + ni]     = stencil[ 0, +1] # (i  , j+1)
  A[idx, idx + ni - 1] = stencil[+1, +1] # (i+1, j+1)
  #! format: on
end

ilo = 1;
ihi = ni;
jlo = 1;
jhi = nj;
for idx in LI
  i, j = CI[idx].I
  A[idx, idx] = stencil[+0, +0] # (i  , j  )

  # if i >= ilo && j >= jlo # (i-1, j-1)
  im1_jm1 = idx - ni - 1
  # @show im1_jm1
  if 1 <= (idx - ni - 1) <= len # (i-1, j-1)
    A[idx, idx - ni - 1] = stencil[-1, -1]
    # else
    # A[idx, idx - ni - 1] = 0
  end

  # if j >= jlo  # (i  , j-1)
  if 1 <= (idx - ni) < len  # (i  , j-1)
    A[idx, idx - ni] = stencil[+0, -1]
    # else
    #   A[idx, idx - ni] = 0
  end

  # if i <= ihi && j >= jlo # (i+1, j-1)
  if 1 <= (idx - ni + 1) <= len # (i+1, j-1)
    A[idx, idx - ni + 1] = stencil[+1, -1]
    # else
    #   A[idx, idx - ni + 1] = 0
  end

  if 1 <= (idx - 1) <= len # (i-1, j  )
    A[idx, idx - 1] = stencil[-1, +0]
    # else
    #   A[idx, idx - 1] = 0
  end

  # if i <= ihi # (i+1, j  )
  if 1 <= idx + 1 <= len # (i+1, j  )
    A[idx, idx + 1] = stencil[+1, +0]
    # else
    #   A[idx, idx + 1] = 0
  end

  # if i >= ilo && j <= jhi # (i-1, j+1)
  if 1 <= (idx + ni + 1) <= len # (i-1, j+1)
    A[idx, idx + ni + 1] = stencil[-1, +1]
    # else
    #   A[idx, idx + ni + 1] = 0
  end

  # if j <= jhi # (i  , j+1)
  if 1 <= idx + ni <= len # (i  , j+1)
    A[idx, idx + ni] = stencil[0, +1]
    # else
    #   A[idx, idx + ni] = 0
  end

  # if i <= ihi && j <= jhi # (i+1, j+1)
  if 1 <= idx + ni - 1 <= len # (i+1, j+1)
    A[idx, idx + ni - 1] = stencil[+1, +1]
    # else
    #   A[idx, idx + ni - 1] = 0
  end
end

# # A = poisson(len);
b = rand(len);
# x = zeros(len)

prob = LinearProblem(A, b)
sol = solve(prob)
Pl = ilu0(A);
solve(prob, KrylovJL_GMRES(); Pl=Pl);

@btime begin
  solve($prob, $KrylovJL_GMRES(); Pl=$Pl)
end

@btime begin
  solve($prob, $KrylovJL_CG(); Pl=$Pl)
end

# x = gmres(A, b; verbose=10);

# ml = ruge_stuben(A) # Construct a Ruge-Stuben solver

# amg = AlgebraicMultigrid._solve(ml, b) # should return ones(1000)

# smg = solve(A, b, RugeStubenAMG(); maxiter=100, abstol=1e-6, verbose=true);
