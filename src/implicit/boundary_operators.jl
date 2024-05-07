
abstract type AbstractBC end

struct DirichletBC{T} <: AbstractBC
  val::T
end

struct NeumannBC <: AbstractBC end

const ILO_BC_LOC = 1
const IHI_BC_LOC = 2
const JLO_BC_LOC = 3
const JHI_BC_LOC = 4
const KLO_BC_LOC = 5
const KHI_BC_LOC = 6

# applybc!(::NeumannBC, u, mesh, loc::Int) = nothing

function applybcs!(bcs, mesh, u::AbstractArray)
  for (i, bc) in enumerate(bcs)
    applybc!(bc, mesh, u, i)
  end
end

function applybc!(::NeumannBC, mesh::CurvilinearGrid2D, u::AbstractArray, loc::Int)
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

function applybc!(::NeumannBC, mesh::CurvilinearGrid3D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC
      copy!(u[ilo - 1, :, :], u[ilo, :, :])
    elseif loc == IHI_BC_LOC
      copy!(u[ihi + 1, :, :], u[ihi, :, :])
    elseif loc == JLO_BC_LOC
      copy!(u[:, jlo - 1, :], u[:, jlo, :])
    elseif loc == JHI_BC_LOC
      copy!(u[:, jhi + 1, :], u[:, jhi, :])
    elseif loc == KLO_BC_LOC
      copy!(u[:, :, klo - 1], u[:, :, klo])
    elseif loc == KHI_BC_LOC
      copy!(u[:, :, khi + 1], u[:, :, khi])
    else
      error("Bad 3d boundary location value $(loc), must be 1-6")
    end
  end
end

#
# function applybc!(bc::DirichletBC, ::CurvilinearGrid1D, u::AbstractArray, limits, loc::Int)
#   @unpack ilo, ihi = mesh.domain_limits.cell

#   @views begin
#     if loc == ILO_BC_LOC
#       u[ilo - 1] .= bc.val
#     elseif loc == IHI_BC_LOC
#       u[ihi + 1] .= bc.val
#     else
#       error("Bad 1d boundary location value $(loc), must be 1 or 2")
#     end
#   end
# end

function applybc!(bc::DirichletBC, mesh::CurvilinearGrid2D, u::AbstractArray, loc::Int)
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

function applybc!(bc::DirichletBC, mesh::CurvilinearGrid3D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  # @show ilo, ihi, jlo, jhi, klo, khi, loc, bc.val
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

function bc_coeffs(::NeumannBC, ::CartesianIndex{2}, loc, T)
  Aᵢ₋₁ⱼ₋₁ = Aᵢⱼ₋₁ = Aᵢ₊₁ⱼ₋₁ = Aᵢ₋₁ⱼ = Aᵢ₊₁ⱼ = Aᵢ₋₁ⱼ₊₁ = Aᵢⱼ₊₁ = Aᵢ₊₁ⱼ₊₁ = zero(T)
  Aᵢⱼ = one(T)

  if loc == ILO_BC_LOC
    Aᵢ₊₁ⱼ = -one(T)
  elseif loc == IHI_BC_LOC
    Aᵢ₋₁ⱼ = -one(T)
  elseif loc == JLO_BC_LOC
    Aᵢⱼ₊₁ = -one(T)
  elseif loc == JHI_BC_LOC
    Aᵢⱼ₋₁ = -one(T)
  else
    error("Bad boundary location! ($loc)")
  end

  A_coeffs = SVector(Aᵢ₋₁ⱼ₋₁, Aᵢⱼ₋₁, Aᵢ₊₁ⱼ₋₁, Aᵢ₋₁ⱼ, Aᵢⱼ, Aᵢ₊₁ⱼ, Aᵢ₋₁ⱼ₊₁, Aᵢⱼ₊₁, Aᵢ₊₁ⱼ₊₁)
  rhs_coeff = zero(T)
  return A_coeffs, rhs_coeff
end

function bc_coeffs(::NeumannBC, ::CartesianIndex{3}, loc, T)
  Aᵢ₋₁ⱼ₋₁ₖ₋₁ = zero(T)
  Aᵢⱼ₋₁ₖ₋₁ = zero(T)
  Aᵢ₊₁ⱼ₋₁ₖ₋₁ = zero(T)
  Aᵢ₋₁ⱼₖ₋₁ = zero(T)
  Aᵢⱼₖ₋₁ = zero(T)
  Aᵢ₊₁ⱼₖ₋₁ = zero(T)
  Aᵢ₋₁ⱼ₊₁ₖ₋₁ = zero(T)
  Aᵢⱼ₊₁ₖ₋₁ = zero(T)
  Aᵢ₊₁ⱼ₊₁ₖ₋₁ = zero(T)
  Aᵢ₋₁ⱼ₋₁ₖ = zero(T)
  Aᵢⱼ₋₁ₖ = zero(T)
  Aᵢ₊₁ⱼ₋₁ₖ = zero(T)
  Aᵢ₋₁ⱼₖ = zero(T)
  Aᵢⱼₖ = one(T)
  Aᵢ₊₁ⱼₖ = zero(T)
  Aᵢ₋₁ⱼ₊₁ₖ = zero(T)
  Aᵢⱼ₊₁ₖ = zero(T)
  Aᵢ₊₁ⱼ₊₁ₖ = zero(T)
  Aᵢ₋₁ⱼ₋₁ₖ₊₁ = zero(T)
  Aᵢⱼ₋₁ₖ₊₁ = zero(T)
  Aᵢ₊₁ⱼ₋₁ₖ₊₁ = zero(T)
  Aᵢ₋₁ⱼₖ₊₁ = zero(T)
  Aᵢⱼₖ₊₁ = zero(T)
  Aᵢ₊₁ⱼₖ₊₁ = zero(T)
  Aᵢ₋₁ⱼ₊₁ₖ₊₁ = zero(T)
  Aᵢⱼ₊₁ₖ₊₁ = zero(T)
  Aᵢ₊₁ⱼ₊₁ₖ₊₁ = zero(T)

  if loc == ILO_BC_LOC
    Aᵢ₊₁ⱼₖ = -one(T)
  elseif loc == IHI_BC_LOC
    Aᵢ₋₁ⱼₖ = -one(T)
  elseif loc == JLO_BC_LOC
    Aᵢⱼ₊₁ₖ = -one(T)
  elseif loc == JHI_BC_LOC
    Aᵢⱼ₋₁ₖ = -one(T)
  elseif loc == KLO_BC_LOC
    Aᵢⱼₖ₊₁ = -one(T)
  elseif loc == KHI_BC_LOC
    Aᵢⱼₖ₋₁ = -one(T)
  else
    error("Bad boundary location! ($loc)")
  end

  A_coeffs = SVector(
    Aᵢ₋₁ⱼ₋₁ₖ₋₁,
    Aᵢⱼ₋₁ₖ₋₁,
    Aᵢ₊₁ⱼ₋₁ₖ₋₁,
    Aᵢ₋₁ⱼₖ₋₁,
    Aᵢⱼₖ₋₁,
    Aᵢ₊₁ⱼₖ₋₁,
    Aᵢ₋₁ⱼ₊₁ₖ₋₁,
    Aᵢⱼ₊₁ₖ₋₁,
    Aᵢ₊₁ⱼ₊₁ₖ₋₁,
    Aᵢ₋₁ⱼ₋₁ₖ,
    Aᵢⱼ₋₁ₖ,
    Aᵢ₊₁ⱼ₋₁ₖ,
    Aᵢ₋₁ⱼₖ,
    Aᵢⱼₖ,
    Aᵢ₊₁ⱼₖ,
    Aᵢ₋₁ⱼ₊₁ₖ,
    Aᵢⱼ₊₁ₖ,
    Aᵢ₊₁ⱼ₊₁ₖ,
    Aᵢ₋₁ⱼ₋₁ₖ₊₁,
    Aᵢⱼ₋₁ₖ₊₁,
    Aᵢ₊₁ⱼ₋₁ₖ₊₁,
    Aᵢ₋₁ⱼₖ₊₁,
    Aᵢⱼₖ₊₁,
    Aᵢ₊₁ⱼₖ₊₁,
    Aᵢ₋₁ⱼ₊₁ₖ₊₁,
    Aᵢⱼ₊₁ₖ₊₁,
    Aᵢ₊₁ⱼ₊₁ₖ₊₁,
  )
  rhs_coeff = zero(T)
  return A_coeffs, rhs_coeff
end

function bc_coeffs(bc::DirichletBC, ::CartesianIndex{2}, loc, T)
  A_coeffs = SVector{9,T}(0, 0, 0, 0, 1, 0, 0, 0, 0)
  rhs_coeff = bc.val
  return A_coeffs, rhs_coeff
end

function bc_coeffs(bc::DirichletBC, ::CartesianIndex{3}, loc, T)
  A_coeffs = SVector{27,T}(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  )
  rhs_coeff = bc.val
  return A_coeffs, rhs_coeff
end
