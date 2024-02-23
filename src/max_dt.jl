module TimeStepControl

using UnPack
using CurvilinearGrids

export max_dt

function max_dt(diffusivity, mesh::CurvilinearGrid2D)
  maxdt = Inf

  @inline for idx in mesh.iterators.cell.domain
    # convert from linear index to cartesian tuple and
    # then add 1/2 to get the centroid indices
    centroid_idx = idx.I .+ 0.5
    @unpack ξx, ξy, ηx, ηy = metrics(mesh, centroid_idx)

    dξ = 1 / sqrt(mesh.cell_center_metrics.ξ.x[idx]^2 + #
                  mesh.cell_center_metrics.ξ.y[idx]^2)
    dη = 1 / sqrt(mesh.cell_center_metrics.η.x[idx]^2 + #
                  mesh.cell_center_metrics.η.y[idx]^2)

    α = diffusivity[idx]

    dt = min(
      (dξ^2) / α, # ξ direction
      (dη^2) / α, # η direction
    )

    maxdt = min(dt, maxdt)
  end

  return maxdt
end

end
