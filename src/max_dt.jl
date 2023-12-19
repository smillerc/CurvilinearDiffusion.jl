module TimeStepControl

using UnPack
using CurvilinearGrids

export max_dt

function max_dt(solver, mesh::CurvilinearGrid2D)
  maxdt = Inf
  @unpack ilo, ihi, jlo, jhi = mesh.limits
  CI = CartesianIndices((ilo:ihi, jlo:jhi))

  @inline for idx in CI
    # convert from linear index to cartesian tuple and
    # then add 1/2 to get the centroid indices
    centroid_idx = idx.I .+ 0.5
    @unpack ξx, ξy, ηx, ηy = metrics(mesh, centroid_idx)

    dξ = 1 / sqrt(ξx^2 + ξy^2)
    dη = 1 / sqrt(ηx^2 + ηy^2)

    α = solver.aⁿ⁺¹[idx]

    dt = min(
      (dξ^2) / α, # ξ direction
      (dη^2) / α, # η direction
    )

    maxdt = min(dt, maxdt)
  end

  return maxdt
end

end
