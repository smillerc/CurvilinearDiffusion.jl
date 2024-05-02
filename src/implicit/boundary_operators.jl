
abstract type AbstractBC end

struct DirichletBC{T} <: AbstractBC
  val::T
end

struct NeumannBC <: AbstractBC end

bc_operator(::NeumannBC) = _neumann_boundary_diffusion_operator
bc_operator(::DirichletBC) = _dirichlet_boundary_diffusion_operator

const ILO_BC_LOC = 1
const IHI_BC_LOC = 2
const JLO_BC_LOC = 3
const JHI_BC_LOC = 4
const KLO_BC_LOC = 5
const KHI_BC_LOC = 6

applybc!(::NeumannBC, u, mesh, loc::Int) = nothing

function applybc!(bcs, u, mesh)
  for (i, bc) in enumerate(bcs)
    applybc!(bc, u, mesh, i)
  end
end

function applybc!(::NeumannBC, u, mesh::CurvilinearGrid2D, loc::Int)
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      copy!(u[ilo, :], u[ilo - 1, :])
    elseif loc == IHI_BC_LOC
      copy!(u[ihi, :], u[ihi + 1, :])
    elseif loc == JLO_BC_LOC
      copy!(u[:, jlo], u[:, jlo - 1])
    elseif loc == JHI_BC_LOC
      copy!(u[:, jhi], u[:, jhi + 1])
    else
      error("Bad 2d boundary location value $(loc), must be 1-4")
    end
  end
end

# 
function applybc!(bc::DirichletBC, u, mesh::CurvilinearGrid1D, loc::Int)
  @unpack ilo, ihi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1] .= bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1] .= bc.val
    else
      error("Bad 1d boundary location value $(loc), must be 1 or 2")
    end
  end
end

function applybc!(bc::DirichletBC, u, mesh::CurvilinearGrid2D, loc::Int)
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1, :] .= bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1, :] .= bc.val
    elseif loc == JLO_BC_LOC
      u[:, jlo - 1] .= bc.val
    elseif loc == JHI_BC_LOC
      u[:, jhi + 1] .= bc.val
    else
      error("Bad 2d boundary location value $(loc), must be 1-4")
    end
  end
end

function applybc!(bc::DirichletBC, u, mesh::CurvilinearGrid3D, loc::Int)
  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1, :, :] .= bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1, :, :] .= bc.val
    elseif loc == JLO_BC_LOC
      u[:, jlo - 1, :] .= bc.val
    elseif loc == JHI_BC_LOC
      u[:, jhi + 1, :] .= bc.val
    elseif loc == KLO_BC_LOC
      u[:, :, klo - 1] .= bc.val
    elseif loc == KHI_BC_LOC
      u[:, :, khi + 1] .= bc.val
    else
      error("Bad 3d boundary location value $(loc), must be 1-6")
    end
  end
end

# struct RobinBC <: AbstractBC end

# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------

@kernel function boundary_diffusion_op_kernel_2d!(
  A,
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F1,
  stencil_col_lookup,
  boundary_operator::BC,
  loc::Int,
) where {F1<:Function,BC}
  idx = @index(Global, Linear)

  _, ncols = size(A)
  @inbounds begin
    grid_idx = grid_indices[idx]
    row = matrix_indices[idx]

    edge_α = edge_diffusivity(α, grid_idx, mean_func)
    J = cell_center_metrics.J[grid_idx]
    stencil = boundary_operator(edge_α, Δt, J, edge_metrics, grid_idx, loc)

    ᵢ₋₁ = !(loc == ILO_BC_LOC)
    ᵢ₊₁ = !(loc == IHI_BC_LOC)
    ⱼ₋₁ = !(loc == JLO_BC_LOC)
    ⱼ₊₁ = !(loc == JHI_BC_LOC)

    #! format: off
    colᵢ₋₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁)
    colᵢⱼ₋₁ =   row + first(stencil_col_lookup.ᵢⱼ₋₁)
    colᵢ₊₁ⱼ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁)
    colᵢ₋₁ⱼ =   row + first(stencil_col_lookup.ᵢ₋₁ⱼ)
    colᵢⱼ =     row + first(stencil_col_lookup.ᵢⱼ)
    colᵢ₊₁ⱼ =   row + first(stencil_col_lookup.ᵢ₊₁ⱼ)
    colᵢ₋₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁)
    colᵢⱼ₊₁ =   row + first(stencil_col_lookup.ᵢⱼ₊₁)
    colᵢ₊₁ⱼ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁)

    if ((ᵢ₋₁ && ⱼ₋₁) && (1 <= colᵢ₋₁ⱼ₋₁ <= ncols)) A[row, colᵢ₋₁ⱼ₋₁] = stencil[1] end # (i-1, j-1)
    if ((       ⱼ₋₁) && (1 <= colᵢⱼ₋₁   <= ncols)) A[row, colᵢⱼ₋₁  ] = stencil[2] end # (i  , j-1)
    if ((ᵢ₋₁       ) && (1 <= colᵢ₊₁ⱼ₋₁ <= ncols)) A[row, colᵢ₊₁ⱼ₋₁] = stencil[3] end # (i+1, j-1)
    if ((ᵢ₋₁ && ⱼ₊₁) && (1 <= colᵢ₋₁ⱼ   <= ncols)) A[row, colᵢ₋₁ⱼ  ] = stencil[4] end # (i-1, j  )
    
    A[row, colᵢⱼ] = stencil[5]  #[+0, +0] # (i  , j  )

    if ((       ⱼ₊₁) && (1 <= colᵢ₊₁ⱼ   <= ncols)) A[row, colᵢ₊₁ⱼ  ] = stencil[6] end # (i+1, j  )
    if ((ᵢ₊₁ && ⱼ₋₁) && (1 <= colᵢ₋₁ⱼ₊₁ <= ncols)) A[row, colᵢ₋₁ⱼ₊₁] = stencil[7] end # (i-1, j+1)
    if ((ᵢ₊₁       ) && (1 <= colᵢⱼ₊₁   <= ncols)) A[row, colᵢⱼ₊₁  ] = stencil[8] end # (i  , j+1)
    if ((ᵢ₊₁ && ⱼ₊₁) && (1 <= colᵢ₊₁ⱼ₊₁ <= ncols)) A[row, colᵢ₊₁ⱼ₊₁] = stencil[9] end # (i+1, j+1)
    #! format: on

  end
end

@kernel function boundary_diffusion_op_kernel_3d!(
  A,
  α,
  Δt,
  cell_center_metrics,
  edge_metrics,
  grid_indices,
  matrix_indices,
  mean_func::F1,
  stencil_col_lookup,
  boundary_operator::F2,
  loc::Int,
) where {F1<:Function,F2<:Function}
  idx = @index(Global, Linear)

  _, ncols = size(A)
  @inbounds begin
    grid_idx = grid_indices[idx]
    row = matrix_indices[idx]

    edge_α = edge_diffusivity(α, grid_idx, mean_func)
    J = cell_center_metrics.J[grid_idx]
    stencil = boundary_operator(edge_α, Δt, J, edge_metrics, grid_idx, loc)

    ᵢ₊₁ = !(loc == ILO_BC_LOC)
    ⱼ₊₁ = !(loc == IHI_BC_LOC)
    ₖ₊₁ = !(loc == JLO_BC_LOC)
    ᵢ₋₁ = !(loc == JHI_BC_LOC)
    ⱼ₋₁ = !(loc == KLO_BC_LOC)
    ₖ₋₁ = !(loc == KHI_BC_LOC)

    #! format: off
    colᵢ₋₁ⱼ₋₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₋₁)
    colᵢⱼ₋₁ₖ₋₁   = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₋₁)
    colᵢ₊₁ⱼ₋₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₋₁)
    colᵢ₋₁ⱼₖ₋₁   = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₋₁)
    colᵢⱼₖ₋₁     = row + first(stencil_col_lookup.ᵢⱼₖ₋₁)
    colᵢ₊₁ⱼₖ₋₁   = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₋₁)
    colᵢ₋₁ⱼ₊₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₋₁)
    colᵢⱼ₊₁ₖ₋₁   = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₋₁)
    colᵢ₊₁ⱼ₊₁ₖ₋₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₋₁)
    colᵢ₋₁ⱼ₋₁ₖ   = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ)
    colᵢⱼ₋₁ₖ     = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ)
    colᵢ₊₁ⱼ₋₁ₖ   = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ)
    colᵢ₋₁ⱼₖ     = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ)
    colᵢⱼₖ       = row + first(stencil_col_lookup.ᵢⱼₖ)
    colᵢ₊₁ⱼₖ     = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ)
    colᵢ₋₁ⱼ₊₁ₖ   = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ)
    colᵢⱼ₊₁ₖ     = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ)
    colᵢ₊₁ⱼ₊₁ₖ   = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ)
    colᵢ₋₁ⱼ₋₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₋₁ₖ₊₁)
    colᵢⱼ₋₁ₖ₊₁   = row + first(stencil_col_lookup.ᵢⱼ₋₁ₖ₊₁)
    colᵢ₊₁ⱼ₋₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₋₁ₖ₊₁)
    colᵢ₋₁ⱼₖ₊₁   = row + first(stencil_col_lookup.ᵢ₋₁ⱼₖ₊₁)
    colᵢⱼₖ₊₁     = row + first(stencil_col_lookup.ᵢⱼₖ₊₁)
    colᵢ₊₁ⱼₖ₊₁   = row + first(stencil_col_lookup.ᵢ₊₁ⱼₖ₊₁)
    colᵢ₋₁ⱼ₊₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₋₁ⱼ₊₁ₖ₊₁)
    colᵢⱼ₊₁ₖ₊₁   = row + first(stencil_col_lookup.ᵢⱼ₊₁ₖ₊₁)
    colᵢ₊₁ⱼ₊₁ₖ₊₁ = row + first(stencil_col_lookup.ᵢ₊₁ⱼ₊₁ₖ₊₁)

    
    if ((ᵢ₋₁ && ⱼ₋₁ && ₖ₋₁) && (1 <= colᵢ₋₁ⱼ₋₁ₖ₋₁ <= ncols))    A[row, colᵢ₋₁ⱼ₋₁ₖ₋₁ ]  = stencil[1] end
    if ((       ⱼ₋₁ && ₖ₋₁) && (1 <= colᵢⱼ₋₁ₖ₋₁ <= ncols))      A[row, colᵢⱼ₋₁ₖ₋₁   ]  = stencil[2] end
    if ((ᵢ₊₁ && ⱼ₋₁ && ₖ₋₁) && (1 <= colᵢ₊₁ⱼ₋₁ₖ₋₁ <= ncols))    A[row, colᵢ₊₁ⱼ₋₁ₖ₋₁ ]  = stencil[3] end
    if ((ᵢ₋₁        && ₖ₋₁) && (1 <= colᵢ₋₁ⱼₖ₋₁ <= ncols))      A[row, colᵢ₋₁ⱼₖ₋₁   ]  = stencil[4] end
    if ((              ₖ₋₁) && (1 <= colᵢⱼₖ₋₁ <= ncols))        A[row, colᵢⱼₖ₋₁     ]  = stencil[5] end
    if ((ᵢ₊₁        && ₖ₋₁) && (1 <= colᵢ₊₁ⱼₖ₋₁ <= ncols))      A[row, colᵢ₊₁ⱼₖ₋₁   ]  = stencil[6] end
    if ((ᵢ₋₁ && ⱼ₊₁ && ₖ₋₁) && (1 <= colᵢ₋₁ⱼ₊₁ₖ₋₁ <= ncols))    A[row, colᵢ₋₁ⱼ₊₁ₖ₋₁ ]  = stencil[7] end
    if ((       ⱼ₊₁ && ₖ₋₁) && (1 <= colᵢⱼ₊₁ₖ₋₁ <= ncols))      A[row, colᵢⱼ₊₁ₖ₋₁   ]  = stencil[8] end
    if ((ᵢ₊₁ && ⱼ₊₁ && ₖ₋₁) && (1 <= colᵢ₊₁ⱼ₊₁ₖ₋₁ <= ncols))    A[row, colᵢ₊₁ⱼ₊₁ₖ₋₁ ]  = stencil[9] end
    if ((ᵢ₋₁ && ⱼ₋₁       ) && (1 <= colᵢ₋₁ⱼ₋₁ₖ <= ncols))      A[row, colᵢ₋₁ⱼ₋₁ₖ   ]  = stencil[10] end
    if ((       ⱼ₋₁       ) && (1 <= colᵢⱼ₋₁ₖ <= ncols))        A[row, colᵢⱼ₋₁ₖ     ]  = stencil[11] end
    if ((ᵢ₊₁ && ⱼ₋₁       ) && (1 <= colᵢ₊₁ⱼ₋₁ₖ <= ncols))      A[row, colᵢ₊₁ⱼ₋₁ₖ   ]  = stencil[12] end
    if ((ᵢ₋₁              ) && (1 <= colᵢ₋₁ⱼₖ <= ncols))        A[row, colᵢ₋₁ⱼₖ     ]  = stencil[13] end
    
    A[row, colᵢⱼₖ] = stencil[14]

    if ((ᵢ₊₁              ) && (1 <= colᵢ₊₁ⱼₖ <= ncols))        A[row, colᵢ₊₁ⱼₖ     ]  = stencil[15] end
    if ((ᵢ₋₁ && ⱼ₊₁       ) && (1 <= colᵢ₋₁ⱼ₊₁ₖ <= ncols))      A[row, colᵢ₋₁ⱼ₊₁ₖ   ]  = stencil[16] end
    if ((       ⱼ₊₁       ) && (1 <= colᵢⱼ₊₁ₖ <= ncols))        A[row, colᵢⱼ₊₁ₖ     ]  = stencil[17] end
    if ((ᵢ₊₁ && ⱼ₊₁       ) && (1 <= colᵢ₊₁ⱼ₊₁ₖ <= ncols))      A[row, colᵢ₊₁ⱼ₊₁ₖ   ]  = stencil[18] end
    if ((ᵢ₋₁ && ⱼ₋₁ && ₖ₊₁) && (1 <= colᵢ₋₁ⱼ₋₁ₖ₊₁ <= ncols))    A[row, colᵢ₋₁ⱼ₋₁ₖ₊₁ ]  = stencil[19] end
    if ((       ⱼ₋₁ && ₖ₊₁) && (1 <= colᵢⱼ₋₁ₖ₊₁ <= ncols))      A[row, colᵢⱼ₋₁ₖ₊₁   ]  = stencil[20] end
    if ((ᵢ₊₁ && ⱼ₋₁ && ₖ₊₁) && (1 <= colᵢ₊₁ⱼ₋₁ₖ₊₁ <= ncols))    A[row, colᵢ₊₁ⱼ₋₁ₖ₊₁ ]  = stencil[21] end
    if ((ᵢ₋₁        && ₖ₊₁) && (1 <= colᵢ₋₁ⱼₖ₊₁ <= ncols))      A[row, colᵢ₋₁ⱼₖ₊₁   ]  = stencil[22] end
    if ((              ₖ₊₁) && (1 <= colᵢⱼₖ₊₁ <= ncols))        A[row, colᵢⱼₖ₊₁     ]  = stencil[23] end
    if ((ᵢ₊₁        && ₖ₊₁) && (1 <= colᵢ₊₁ⱼₖ₊₁ <= ncols))      A[row, colᵢ₊₁ⱼₖ₊₁   ]  = stencil[24] end
    if ((ᵢ₋₁ && ⱼ₊₁ && ₖ₊₁) && (1 <= colᵢ₋₁ⱼ₊₁ₖ₊₁ <= ncols))    A[row, colᵢ₋₁ⱼ₊₁ₖ₊₁ ]  = stencil[25] end
    if ((       ⱼ₊₁ && ₖ₊₁) && (1 <= colᵢⱼ₊₁ₖ₊₁ <= ncols))      A[row, colᵢⱼ₊₁ₖ₊₁   ]  = stencil[26] end
    if ((ᵢ₊₁ && ⱼ₊₁ && ₖ₊₁) && (1 <= colᵢ₊₁ⱼ₊₁ₖ₊₁ <= ncols))    A[row, colᵢ₊₁ⱼ₊₁ₖ₊₁ ]  = stencil[27] end

    #! format: on

  end
end

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------

# Generate a stencil for a 2D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{2}, loc
)
  T = eltype(edge_diffusivity)

  @unpack a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½ = conservative_edge_terms(
    edge_diffusivity, edge_metrics, idx
  )

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if ILO_BC_LOC == 1
    a_Jξ²ᵢ₋½ = zero(T)
    a_Jξηᵢ₋½ = zero(T)
  elseif IHI_BC_LOC == 2
    a_Jξ²ᵢ₊½ = zero(T)
    a_Jξηᵢ₊½ = zero(T)
  elseif JLO_BC_LOC == 3
    a_Jη²ⱼ₋½ = zero(T)
    a_Jηξⱼ₋½ = zero(T)
  elseif JHI_BC_LOC == 4
    a_Jη²ⱼ₊½ = zero(T)
    a_Jηξⱼ₊½ = zero(T)
  else
    error("bad boundary location")
  end

  edge_terms = (;
    a_Jξ²ᵢ₊½, a_Jξ²ᵢ₋½, a_Jη²ⱼ₊½, a_Jη²ⱼ₋½, a_Jξηᵢ₊½, a_Jξηᵢ₋½, a_Jηξⱼ₊½, a_Jηξⱼ₋½
  )
  stencil = stencil_2d(edge_terms, J, Δτ)

  return stencil
end

# Generate a stencil for a 3D neumann boundary condition
@inline function _neumann_boundary_diffusion_operator(
  edge_diffusivity, Δτ, J, edge_metrics, idx::CartesianIndex{3}, loc
)
  T = eltype(edge_diffusivity)

  edge_terms = conservative_edge_terms(edge_diffusivity, edge_metrics, idx)

  a_Jξ²ᵢ₊½ = edge_terms.a_Jξ²ᵢ₊½
  a_Jξηᵢ₊½ = edge_terms.a_Jξηᵢ₊½
  a_Jξζᵢ₊½ = edge_terms.a_Jξζᵢ₊½

  a_Jξ²ᵢ₋½ = edge_terms.a_Jξ²ᵢ₋½
  a_Jξηᵢ₋½ = edge_terms.a_Jξηᵢ₋½
  a_Jξζᵢ₋½ = edge_terms.a_Jξζᵢ₋½

  a_Jηξⱼ₊½ = edge_terms.a_Jηξⱼ₊½
  a_Jη²ⱼ₊½ = edge_terms.a_Jη²ⱼ₊½
  a_Jηζⱼ₊½ = edge_terms.a_Jηζⱼ₊½

  a_Jηξⱼ₋½ = edge_terms.a_Jηξⱼ₋½
  a_Jη²ⱼ₋½ = edge_terms.a_Jη²ⱼ₋½
  a_Jηζⱼ₋½ = edge_terms.a_Jηζⱼ₋½

  a_Jζξₖ₊½ = edge_terms.a_Jζξₖ₊½
  a_Jζηₖ₊½ = edge_terms.a_Jζηₖ₊½
  a_Jζ²ₖ₊½ = edge_terms.a_Jζ²ₖ₊½

  a_Jζξₖ₋½ = edge_terms.a_Jζξₖ₋½
  a_Jζηₖ₋½ = edge_terms.a_Jζηₖ₋½
  a_Jζ²ₖ₋½ = edge_terms.a_Jζ²ₖ₋½

  # Create a stencil matrix to hold the coefficients for u[i±1,j±1]
  if loc == ILO_BC_LOC
    a_Jξ²ᵢ₋½ = a_Jξηᵢ₋½ = a_Jξζᵢ₋½ = zero(T)

  elseif loc == IHI_BC_LOC
    a_Jξ²ᵢ₊½ = a_Jξηᵢ₊½ = a_Jξζᵢ₊½ = zero(T)

  elseif loc == JLO_BC_LOC
    a_Jηξⱼ₋½ = a_Jη²ⱼ₋½ = a_Jηζⱼ₋½ = zero(T)

  elseif loc == JHI_BC_LOC
    a_Jηξⱼ₊½ = a_Jη²ⱼ₊½ = a_Jηζⱼ₊½ = zero(T)

  elseif loc == KLO_BC_LOC
    a_Jζξₖ₋½ = a_Jζηₖ₋½ = a_Jζ²ₖ₋½ = zero(T)

  elseif loc == KHI_BC_LOC
    a_Jζξₖ₊½ = a_Jζηₖ₊½ = a_Jζ²ₖ₊½ = zero(T)
  else
    error("bad boundary location")
  end

  edge_terms = (;
    a_Jξ²ᵢ₊½,
    a_Jξ²ᵢ₋½,
    a_Jξηᵢ₊½,
    a_Jξηᵢ₋½,
    a_Jξζᵢ₊½,
    a_Jξζᵢ₋½,
    a_Jηξⱼ₊½,
    a_Jηξⱼ₋½,
    a_Jη²ⱼ₊½,
    a_Jη²ⱼ₋½,
    a_Jηζⱼ₊½,
    a_Jηζⱼ₋½,
    a_Jζξₖ₊½,
    a_Jζξₖ₋½,
    a_Jζηₖ₊½,
    a_Jζηₖ₋½,
    a_Jζ²ₖ₊½,
    a_Jζ²ₖ₋½,
  )

  stencil = stencil_3d(edge_terms, J, Δτ)

  return stencil
end