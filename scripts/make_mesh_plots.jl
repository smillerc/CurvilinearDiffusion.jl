using CairoMakie
using CurvilinearGrids

function wavy_grid_1(nx, ny, nhalo=1)
  x2d = zeros(nx, ny)
  y2d = zeros(nx, ny)

  x1d = range(0, 1; length=nx)
  y1d = range(0, 1; length=ny)
  a0 = 0.1
  for I in CartesianIndices(x2d)
    i, j = I.I

    x = x1d[i]
    y = y1d[j]

    x2d[i, j] = x + a0 * sinpi(2x) * sinpi(2y)
    y2d[i, j] = y + a0 * sinpi(2x) * sinpi(2y)

    # x2d[i, j] = ((i - 1) / (nx - 1)) + a0 * sinpi(2x) * sinpi(2y)
    # y2d[i, j] = ((j - 1) / (ny - 1)) + a0 * sinpi(2x) * sinpi(2y)
  end

  mesh = CurvilinearGrids.CurvilinearGrid2D(x2d, y2d, nhalo)
  return mesh
end

function wavy_grid_2(ni, nj, nhalo=1)
  Lx = 12
  Ly = 12
  n_xy = 6
  n_yx = 6

  xmin = -Lx / 2
  ymin = -Ly / 2

  Δx0 = Lx / (ni - 1)
  Δy0 = Ly / (nj - 1)

  Ax = 0.2 / Δx0
  Ay = 0.4 / Δy0

  x = zeros(ni, nj)
  y = zeros(ni, nj)
  for j in 1:nj
    for i in 1:ni
      x[i, j] = xmin + Δx0 * ((i - 1) + Ax * sinpi((n_xy * (j - 1) * Δy0) / Ly))
      y[i, j] = ymin + Δy0 * ((j - 1) + Ay * sinpi((n_yx * (i - 1) * Δx0) / Lx))
    end
  end

  return CurvilinearGrid2D(x, y, nhalo)
end

function uniform_grid(nx, ny, nhalo=1)
  x0, x1 = (0, 1)
  y0, y1 = (0, 1)

  mesh = CurvilinearGrids.RectlinearGrid((x0, y0), (x1, y1), (nx, ny), nhalo)
  return mesh
end

begin
  nx = ny = 50
  for (grid_id, grid) in enumerate((
    wavy_grid_1(nx, ny), #
    wavy_grid_2(nx, ny), #
    uniform_grid(nx, ny),#
  ))
    f = Figure(; size=(500, 500))
    ax = Axis(
      f[1, 1]; aspect=1, xlabel="x", ylabel="y", xgridvisible=false, ygridvisible=false
    )

    x, y = coords(grid)

    for (_x, _y) in zip(eachcol(x), eachcol(y))
      lines!(_x, _y; color=:black, linewidth=1)
    end

    for (_x, _y) in zip(eachrow(x), eachrow(y))
      lines!(_x, _y; color=:black, linewidth=1)
    end

    display(f)
    save("mesh$(grid_id).eps", f)
  end
end