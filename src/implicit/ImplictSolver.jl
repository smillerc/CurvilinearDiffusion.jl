module ImplicitSolverType

using LinearSolve

struct ImplicitSolver{SM,V,ST}
  A::SM # sparse matrix
  x::V # solution vector
  solver::ST # linear solver, e.g. GMRES, CG, etc.
end

function ImplicitSolver((ni, nj), solver=KrylovJL_GMRES())
  len = ni * nj

  #! format: off
  A = spdiagm(
    -ni - 1 => ones(len - ni - 1), # (i-1, j-1)
    -ni     => ones(len - ni),     # (i  , j-1)
    -ni + 1 => ones(len - ni + 1), # (i+1, j-1)
    -1      => ones(len - 1),      # (i-1, j  )
    0       => ones(len),          # (i  , j  )
    1       => ones(len - 1),      # (i+1, j  )
    ni - 1  => ones(len - ni + 1), # (i-1, j+1)
    ni      => ones(len - ni),     # (i  , j+1)
    ni + 1  => ones(len - ni - 1), # (i+1, j+1)
  )
  #! format: on

  x = zeros(ni, nj)
  return ImplicitSolver(A, x, solver)
end

end
