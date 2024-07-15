module VTKOutput

using ..ImplicitSchemeType: ImplicitScheme
using ..PseudoTransientScheme: PseudoTransientSolver

using WriteVTK, CurvilinearGrids, Printf

export save_vtk

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)

function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

function save_vtk(scheme, u, mesh, iteration=0, t=0.0, name="diffusion", T=Float32)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"

  mdomain = mesh.iterators.cell.domain
  ddomain = scheme.iterators.domain.cartesian

  coords = Array{T}.(CurvilinearGrids.coords(mesh))

  @views vtk_grid(fn, coords...) do vtk
    vtk["TimeValue"] = t
    vtk["u"] = Array{T}(u[mdomain])
    vtk["diffusivity"] = Array{T}(scheme.α[ddomain])
    vtk["source_term"] = Array{T}(scheme.source_term[ddomain])
  end
end

function save_vtk(
  scheme::PseudoTransientSolver, u, mesh, iteration=0, t=0.0, name="diffusion", T=Float32
)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"

  domain = mesh.iterators.cell.domain

  coords = Array{T}.(CurvilinearGrids.coords(mesh))

  @views vtk_grid(fn, coords...) do vtk
    vtk["TimeValue"] = t
    vtk["u"] = Array{T}(u[domain])
    vtk["u_prev"] = Array{T}(scheme.u_prev[domain])
    vtk["qi"] = Array{T}(scheme.q[1][domain])
    vtk["qj"] = Array{T}(scheme.q[2][domain])
    vtk["q2i"] = Array{T}(scheme.q′[1][domain])
    vtk["q2j"] = Array{T}(scheme.q′[2][domain])

    vtk["residual"] = Array{T}(scheme.residual[domain])
    vtk["Reynolds_number"] = Array{T}(scheme.Reynolds_number[domain])

    vtk["diffusivity"] = Array{T}(scheme.α[domain])

    vtk["dτ_ρ"] = Array{T}(scheme.dτ_ρ[domain])
    vtk["θr_dτ"] = Array{T}(scheme.θr_dτ[domain])

    vtk["source_term"] = Array{T}(scheme.source_term[domain])
  end
end

end
