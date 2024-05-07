module VTKOutput

using ..ImplicitSchemeType: ImplicitScheme

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

function save_vtk(
  scheme::ImplicitScheme, u, mesh, iteration=0, t=0.0, name="diffusion", T=Float32
)
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

end
