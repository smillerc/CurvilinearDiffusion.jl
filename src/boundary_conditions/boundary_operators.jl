module BoundaryConditions

using CurvilinearGrids, UnPack

export DirichletBC, NeumannBC, PeriodicBC, applybc!, applybcs!, check_diffusivity_validity

abstract type AbstractBC end

struct DirichletBC{T} <: AbstractBC
  val::Vector{T}
  is_const::Bool
end

DirichletBC(v::Real) = DirichletBC([v], true)
DirichletBC(v::Real, is_const) = DirichletBC([v], is_const)

struct NeumannBC <: AbstractBC end
struct PeriodicBC <: AbstractBC end

const ILO_BC_LOC = 1
const IHI_BC_LOC = 2
const JLO_BC_LOC = 3
const JHI_BC_LOC = 4
const KLO_BC_LOC = 5
const KHI_BC_LOC = 6

function applybcs!(bcs, mesh, u::AbstractArray)
  for (i, bc) in enumerate(bcs)
    applybc!(bc, mesh, u, i)
  end
end

function applybc!(::NeumannBC, mesh::CurvilinearGrid1D, u::AbstractVector, loc::Int)
  @unpack ilo, ihi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1] = u[ilo]
    elseif loc == IHI_BC_LOC
      u[ihi + 1] = u[ihi]
    else
      error("Bad 1d boundary location value $(loc), must be 1 or 2")
    end
  end
end

function applybc!(::NeumannBC, mesh::CurvilinearGrid2D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC
      copy!(u[ilo - 1, jlo:jhi], u[ilo, jlo:jhi])
    elseif loc == IHI_BC_LOC
      copy!(u[ihi + 1, jlo:jhi], u[ihi, jlo:jhi])
    elseif loc == JLO_BC_LOC
      copy!(u[ilo:ihi, jlo - 1], u[ilo:ihi, jlo])
    elseif loc == JHI_BC_LOC
      copy!(u[ilo:ihi, jhi + 1], u[ilo:ihi, jhi])
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
      copy!(u[ilo - 1, jlo:jhi, klo:khi], u[ilo, jlo:jhi, klo:khi])
    elseif loc == IHI_BC_LOC
      copy!(u[ihi + 1, jlo:jhi, klo:khi], u[ihi, jlo:jhi, klo:khi])
    elseif loc == JLO_BC_LOC
      copy!(u[ilo:ihi, jlo - 1, klo:khi], u[ilo:ihi, jlo, klo:khi])
    elseif loc == JHI_BC_LOC
      copy!(u[ilo:ihi, jhi + 1, klo:khi], u[ilo:ihi, jhi, klo:khi])
    elseif loc == KLO_BC_LOC
      copy!(u[ilo:ihi, jlo:jhi, klo - 1], u[ilo:ihi, jlo:jhi, klo])
    elseif loc == KHI_BC_LOC
      copy!(u[ilo:ihi, jlo:jhi, khi + 1], u[ilo:ihi, jlo:jhi, khi])
    else
      error("Bad 3d boundary location value $(loc), must be 1-6")
    end
  end
end

function applybc!(bc::DirichletBC, mesh::CurvilinearGrid1D, u::AbstractVector, loc::Int)
  @unpack ilo, ihi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1] = bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1] = bc.val
    else
      error("Bad 1d boundary location value $(loc), must be 1 or 2")
    end
  end
end

function applybc!(bc::DirichletBC, mesh::CurvilinearGrid2D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1, jlo:jhi] .= bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1, jlo:jhi] .= bc.val
    elseif loc == JLO_BC_LOC
      u[ilo:ihi, jlo - 1] .= bc.val
    elseif loc == JHI_BC_LOC
      u[ilo:ihi, jhi + 1] .= bc.val
    else
      error("Bad 2d boundary location value $(loc), must be 1-4")
    end
  end
end

function applybc!(bc::DirichletBC, mesh::CurvilinearGrid3D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  @views begin
    if loc == ILO_BC_LOC
      u[ilo - 1, jlo:jhi, klo:khi] .= bc.val
    elseif loc == IHI_BC_LOC
      u[ihi + 1, jlo:jhi, klo:khi] .= bc.val
    elseif loc == JLO_BC_LOC
      u[ilo:ihi, jlo - 1, klo:khi] .= bc.val
    elseif loc == JHI_BC_LOC
      u[ilo:ihi, jhi + 1, klo:khi] .= bc.val
    elseif loc == KLO_BC_LOC
      u[ilo:ihi, jlo:jhi, klo - 1] .= bc.val
    elseif loc == KHI_BC_LOC
      u[ilo:ihi, jlo:jhi, khi + 1] .= bc.val
    else
      error("Bad 3d boundary location value $(loc), must be 1-6")
    end
  end
end

function applybc!(::PeriodicBC, mesh::CurvilinearGrid1D, u::AbstractVector, loc::Int)
  @unpack ilo, ihi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC #|| loc == IHI_BC_LOC
      u[ilo - 1] = u[ihi]
      u[ihi + 1] = u[ilo]
      # else
      #   error("Bad 1d boundary location value $(loc), must be 1 or 2")
    end
  end
end

function applybc!(::PeriodicBC, mesh::CurvilinearGrid2D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC # || loc == IHI_BC_LOC
      copy!(u[ilo - 1, jlo:jhi], u[ihi, jlo:jhi])
      copy!(u[ihi + 1, jlo:jhi], u[ilo, jlo:jhi])
    elseif loc == JLO_BC_LOC # || loc == JHI_BC_LOC
      copy!(u[ilo:ihi, jlo - 1], u[ilo:ihi, jhi])
      copy!(u[ilo:ihi, jhi + 1], u[ilo:ihi, jlo])
      # else
      #   error("Bad 2d boundary location value $(loc), must be 1-4")
    end
  end
end

function applybc!(::PeriodicBC, mesh::CurvilinearGrid3D, u::AbstractArray, loc::Int)
  @unpack ilo, ihi, jlo, jhi, klo, khi = mesh.domain_limits.cell

  # Neumann BCs set the ghost region to be the same as the inner region along the edge,
  # such that the edge diffusivity calculation gets the proper value

  @views begin
    if loc == ILO_BC_LOC # || loc == IHI_BC_LOC
      copy!(u[ilo - 1, jlo:jhi, klo:khi], u[ihi, jlo:jhi, klo:khi])
      copy!(u[ihi + 1, jlo:jhi, klo:khi], u[ilo, jlo:jhi, klo:khi])
    elseif loc == JLO_BC_LOC # || loc == JHI_BC_LOC
      copy!(u[ilo:ihi, jlo - 1, klo:khi], u[ilo:ihi, jhi, klo:khi])
      copy!(u[ilo:ihi, jhi + 1, klo:khi], u[ilo:ihi, jlo, klo:khi])
    elseif loc == KLO_BC_LOC # || loc == KHI_BC_LOC
      copy!(u[ilo:ihi, jlo:jhi, klo - 1], u[ilo:ihi, jlo:jhi, khi])
      copy!(u[ilo:ihi, jlo:jhi, khi + 1], u[ilo:ihi, jlo:jhi, klo])
      # else
      #   error("Bad 3d boundary location value $(loc), must be 1-6")
    end
  end
end

# struct RobinBC <: AbstractBC end

# ---------------------------------------------------------------------------
#  
# ---------------------------------------------------------------------------

bc_rhs_coefficient(::NeumannBC, ::CartesianIndex, T) = zero(T)
bc_rhs_coefficient(::PeriodicBC, ::CartesianIndex, T) = zero(T)
bc_rhs_coefficient(bc::DirichletBC, ::CartesianIndex, T) = bc.val

end