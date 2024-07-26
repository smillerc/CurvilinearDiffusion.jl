
function updatehalo!(Atuple::NamedTuple, nhalo::Int, mpi_topology, do_corners=true)

  # 
  @inbounds for A in Atuple
    updatehalo!(A, nhalo, mpi_topology, do_corners)
  end

  return nothing
end

function updatehalo!(A::AbstractArray{T,1}, nhalo::Int, mpi_topology, ::Bool) where {T}
  domain = expand(CartesianIndices(A), -nhalo)

  iaxis = 1
  ihalo, iedge = haloedge_regions(domain, iaxis, nhalo)

  ilo_neighbor_rank = ilo_neighbor(mpi_topology)
  ihi_neighbor_rank = ihi_neighbor(mpi_topology)

  if ilo_neighbor_rank >= 0 && ihi_neighbor_rank >= 0
    ilo_edge = MPI.Buffer(view(A, iedge.lo))
    ihi_halo = MPI.Buffer(view(A, ihalo.hi))

    MPI.Sendrecv!(
      ilo_edge, # send ilo edge
      ihi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=ilo_neighbor_rank,   # to the ilo rank
      source=ihi_neighbor_rank, # from the ihi rank
    )

    ihi_edge = MPI.Buffer(view(A, iedge.hi))
    ilo_halo = MPI.Buffer(view(A, ihalo.lo))

    MPI.Sendrecv!(
      ihi_edge, # send ihi edge
      ilo_halo, # to ilo halo 
      mpi_topology.comm;
      dest=ihi_neighbor_rank,    # to the ihi rank
      source=ilo_neighbor_rank,  # from the ilo rank
    )
  end

  return nothing
end

NVTX.NVTX.@annotate function updatehalo!(
  A::AbstractArray{T,2}, nhalo::Int, mpi_topology, do_corners=true
) where {T}
  domain = expand(CartesianIndices(A), -nhalo)
  iaxis, jaxis = (1, 2)
  ihalo, iedge = haloedge_regions(domain, iaxis, nhalo)
  jhalo, jedge = haloedge_regions(domain, jaxis, nhalo)

  ilo_neighbor_rank = ilo_neighbor(mpi_topology)
  ihi_neighbor_rank = ihi_neighbor(mpi_topology)
  jlo_neighbor_rank = jlo_neighbor(mpi_topology)
  jhi_neighbor_rank = jhi_neighbor(mpi_topology)

  if ilo_neighbor_rank >= 0 && ihi_neighbor_rank >= 0
    ilo_edge = MPI.Buffer(view(A, iedge.lo))
    ihi_halo = MPI.Buffer(view(A, ihalo.hi))

    MPI.Sendrecv!(
      ilo_edge, # send ilo edge
      ihi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=ilo_neighbor_rank,   # to the ilo rank
      source=ihi_neighbor_rank, # from the ihi rank
    )

    ihi_edge = MPI.Buffer(view(A, iedge.hi))
    ilo_halo = MPI.Buffer(view(A, ihalo.lo))

    MPI.Sendrecv!(
      ihi_edge, # send ihi edge
      ilo_halo, # to ilo halo 
      mpi_topology.comm;
      dest=ihi_neighbor_rank,    # to the ihi rank
      source=ilo_neighbor_rank,  # from the ilo rank
    )
  end

  if jlo_neighbor_rank >= 0 && jhi_neighbor_rank >= 0
    jhi_edge = MPI.Buffer(view(A, jedge.hi))
    jlo_halo = MPI.Buffer(view(A, jhalo.lo))

    MPI.Sendrecv!(
      jhi_edge, # send jhi edge
      jlo_halo, # to jlo halo 
      mpi_topology.comm;
      dest=jhi_neighbor_rank,    # to the jhi rank
      source=jlo_neighbor_rank,  # from the jlo rank
    )

    jlo_edge = MPI.Buffer(view(A, jedge.lo))
    jhi_halo = MPI.Buffer(view(A, jhalo.hi))

    MPI.Sendrecv!(
      jlo_edge, # send ilo edge
      jhi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=jlo_neighbor_rank,    # to the jlo rank
      source=jhi_neighbor_rank,  # from the jhi rank
    )
  end

  if do_corners
    ilojlo_neighbor_rank = neighbor(mpi_topology, -1, -1)
    ihijlo_neighbor_rank = neighbor(mpi_topology, +1, -1)
    ilojhi_neighbor_rank = neighbor(mpi_topology, -1, +1)
    ihijhi_neighbor_rank = neighbor(mpi_topology, +1, +1)

    ilojlo_halo_dom = domain[begin:(begin + nhalo - 1), begin:(begin + nhalo - 1)]
    ilojhi_halo_dom = domain[begin:(begin + nhalo - 1), (end - nhalo + 1):end]
    ihijlo_halo_dom = domain[(end - nhalo + 1):end, begin:(begin + nhalo - 1)]
    ihijhi_halo_dom = domain[(end - nhalo + 1):end, (end - nhalo + 1):end]

    ilojlo_edge_dom = shift(shift(ilojlo_halo_dom, 1, +nhalo), 2, +nhalo)
    ilojhi_edge_dom = shift(shift(ilojhi_halo_dom, 1, +nhalo), 2, -nhalo)
    ihijlo_edge_dom = shift(shift(ihijlo_halo_dom, 1, -nhalo), 2, +nhalo)
    ihijhi_edge_dom = shift(shift(ihijhi_halo_dom, 1, -nhalo), 2, -nhalo)

    if ilojlo_neighbor_rank >= 0 && ihijhi_neighbor_rank >= 0
      ihijhi_edge = MPI.Buffer(view(A, ihijhi_edge_dom))
      ilojlo_halo = MPI.Buffer(view(A, ilojlo_halo_dom))

      MPI.Sendrecv!(
        ihijhi_edge, # send ihi-jhi corner
        ilojlo_halo, # to ilo-jlo halo corner
        mpi_topology.comm;
        dest=ihijhi_neighbor_rank,    # to the ihi-jhi rank
        source=ilojlo_neighbor_rank,  # from the ilo-jlo rank
      )

      ilojlo_edge = MPI.Buffer(view(A, ilojlo_edge_dom))
      ihijhi_halo = MPI.Buffer(view(A, ihijhi_halo_dom))

      MPI.Sendrecv!(
        ilojlo_edge, # send ilo-jlo corner
        ihijhi_halo, # to ihi-jhi halo corner
        mpi_topology.comm;
        dest=ilojlo_neighbor_rank,    # to the ilo-jlo rank
        source=ihijhi_neighbor_rank,  # from the ihi-jhi rank
      )
    end

    if ilojhi_neighbor_rank >= 0 && ihijlo_neighbor_rank >= 0
      ihijlo_edge = MPI.Buffer(view(A, ihijlo_edge_dom))
      ilojhi_halo = MPI.Buffer(view(A, ilojhi_halo_dom))

      MPI.Sendrecv!(
        ihijlo_edge, # send ihi-jlo corner
        ilojhi_halo, # to ilo-jhi halo 
        mpi_topology.comm;
        dest=ihijlo_neighbor_rank,    # to the ihi-jlo rank
        source=ilojhi_neighbor_rank,  # from the ilo-jhi rank
      )

      ilojhi_edge = MPI.Buffer(view(A, ilojhi_edge_dom))
      ihijlo_halo = MPI.Buffer(view(A, ihijlo_halo_dom))

      MPI.Sendrecv!(
        ilojhi_edge, # send ilo-jhi corner
        ihijlo_halo, # to ihi-jlo halo 
        mpi_topology.comm;
        dest=ilojhi_neighbor_rank,    # to the ilo-jhi rank
        source=ihijlo_neighbor_rank,  # from the ihi-jlo rank
      )
    end
  end

  return nothing
end

function updatehalo!(
  A::AbstractArray{T,3}, nhalo::Int, mpi_topology, do_corners=true
) where {T}
  domain = expand(CartesianIndices(A), -nhalo)

  iaxis, jaxis, kaxis = (1, 2, 3)
  ihalo, iedge = haloedge_regions(domain, iaxis, nhalo)
  jhalo, jedge = haloedge_regions(domain, jaxis, nhalo)
  khalo, kedge = haloedge_regions(domain, kaxis, nhalo)

  ilo_neighbor_rank = ilo_neighbor(mpi_topology)
  ihi_neighbor_rank = ihi_neighbor(mpi_topology)
  jlo_neighbor_rank = jlo_neighbor(mpi_topology)
  jhi_neighbor_rank = jhi_neighbor(mpi_topology)
  klo_neighbor_rank = klo_neighbor(mpi_topology)
  khi_neighbor_rank = khi_neighbor(mpi_topology)

  if ilo_neighbor_rank >= 0 && ihi_neighbor_rank >= 0
    ilo_edge = MPI.Buffer(view(A, iedge.lo))
    ihi_halo = MPI.Buffer(view(A, ihalo.hi))

    MPI.Sendrecv!(
      ilo_edge, # send ilo edge
      ihi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=ilo_neighbor_rank,   # to the ilo rank
      source=ihi_neighbor_rank, # from the ihi rank
    )

    ihi_edge = MPI.Buffer(view(A, iedge.hi))
    ilo_halo = MPI.Buffer(view(A, ihalo.lo))

    MPI.Sendrecv!(
      ihi_edge, # send ihi edge
      ilo_halo, # to ilo halo 
      mpi_topology.comm;
      dest=ihi_neighbor_rank,    # to the ihi rank
      source=ilo_neighbor_rank,  # from the ilo rank
    )
  end

  if jlo_neighbor_rank >= 0 && jhi_neighbor_rank >= 0
    jhi_edge = MPI.Buffer(view(A, jedge.hi))
    jlo_halo = MPI.Buffer(view(A, jhalo.lo))

    MPI.Sendrecv!(
      jhi_edge, # send jhi edge
      jlo_halo, # to jlo halo 
      mpi_topology.comm;
      dest=jhi_neighbor_rank,    # to the jhi rank
      source=jlo_neighbor_rank,  # from the jlo rank
    )

    jlo_edge = MPI.Buffer(view(A, jedge.lo))
    jhi_halo = MPI.Buffer(view(A, jhalo.hi))

    MPI.Sendrecv!(
      jlo_edge, # send ilo edge
      jhi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=jlo_neighbor_rank,    # to the jlo rank
      source=jhi_neighbor_rank,  # from the jhi rank
    )
  end

  if klo_neighbor_rank >= 0 && khi_neighbor_rank >= 0
    khi_edge = MPI.Buffer(view(A, kedge.hi))
    klo_halo = MPI.Buffer(view(A, khalo.lo))

    MPI.Sendrecv!(
      khi_edge, # send khi edge
      klo_halo, # to klo halo 
      mpi_topology.comm;
      dest=khi_neighbor_rank,    # to the khi rank
      source=klo_neighbor_rank,  # from the klo rank
    )

    klo_edge = MPI.Buffer(view(A, kedge.lo))
    khi_halo = MPI.Buffer(view(A, khalo.hi))

    MPI.Sendrecv!(
      klo_edge, # send ilo edge
      khi_halo, # to ihi halo 
      mpi_topology.comm;
      dest=klo_neighbor_rank,    # to the klo rank
      source=khi_neighbor_rank,  # from the khi rank
    )
  end

  if do_corners

    # ------

    ilojlo_neighbor_rank = neighbor(mpi_topology, -1, -1, 0)
    ihijlo_neighbor_rank = neighbor(mpi_topology, +1, -1, 0)
    ilojhi_neighbor_rank = neighbor(mpi_topology, -1, +1, 0)
    ihijhi_neighbor_rank = neighbor(mpi_topology, +1, +1, 0)

    ilojlo_halo_dom = domain[begin:(begin + nhalo - 1), begin:(begin + nhalo - 1), :]
    ilojhi_halo_dom = domain[begin:(begin + nhalo - 1), (end - nhalo + 1):end, :]
    ihijlo_halo_dom = domain[(end - nhalo + 1):end, begin:(begin + nhalo - 1), :]
    ihijhi_halo_dom = domain[(end - nhalo + 1):end, (end - nhalo + 1):end, :]

    ilojlo_edge_dom = shift(shift(ilojlo_halo_dom, iaxis, +nhalo), jaxis, +nhalo)
    ilojhi_edge_dom = shift(shift(ilojhi_halo_dom, iaxis, +nhalo), jaxis, -nhalo)
    ihijlo_edge_dom = shift(shift(ihijlo_halo_dom, iaxis, -nhalo), jaxis, +nhalo)
    ihijhi_edge_dom = shift(shift(ihijhi_halo_dom, iaxis, -nhalo), jaxis, -nhalo)

    if ilojlo_neighbor_rank >= 0 && ihijhi_neighbor_rank >= 0
      ihijhi_edge = MPI.Buffer(view(A, ihijhi_edge_dom))
      ilojlo_halo = MPI.Buffer(view(A, ilojlo_halo_dom))

      MPI.Sendrecv!(
        ihijhi_edge, # send ihi-jhi corner
        ilojlo_halo, # to ilo-jlo halo corner
        mpi_topology.comm;
        dest=ihijhi_neighbor_rank,    # to the ihi-jhi rank
        source=ilojlo_neighbor_rank,  # from the ilo-jlo rank
      )

      ilojlo_edge = MPI.Buffer(view(A, ilojlo_edge_dom))
      ihijhi_halo = MPI.Buffer(view(A, ihijhi_halo_dom))

      MPI.Sendrecv!(
        ilojlo_edge, # send ilo-jlo corner
        ihijhi_halo, # to ihi-jhi halo corner
        mpi_topology.comm;
        dest=ilojlo_neighbor_rank,    # to the ilo-jlo rank
        source=ihijhi_neighbor_rank,  # from the ihi-jhi rank
      )
    end

    if ilojhi_neighbor_rank >= 0 && ihijlo_neighbor_rank >= 0
      ihijlo_edge = MPI.Buffer(view(A, ihijlo_edge_dom))
      ilojhi_halo = MPI.Buffer(view(A, ilojhi_halo_dom))

      MPI.Sendrecv!(
        ihijlo_edge, # send ihi-jlo corner
        ilojhi_halo, # to ilo-jhi halo 
        mpi_topology.comm;
        dest=ihijlo_neighbor_rank,    # to the ihi-jlo rank
        source=ilojhi_neighbor_rank,  # from the ilo-jhi rank
      )

      ilojhi_edge = MPI.Buffer(view(A, ilojhi_edge_dom))
      ihijlo_halo = MPI.Buffer(view(A, ihijlo_halo_dom))

      MPI.Sendrecv!(
        ilojhi_edge, # send ilo-jhi corner
        ihijlo_halo, # to ihi-jlo halo 
        mpi_topology.comm;
        dest=ilojhi_neighbor_rank,    # to the ilo-jhi rank
        source=ihijlo_neighbor_rank,  # from the ihi-jlo rank
      )
    end

    # ------

    iloklo_neighbor_rank = neighbor(mpi_topology, -1, 0, -1)
    ihiklo_neighbor_rank = neighbor(mpi_topology, +1, 0, -1)
    ilokhi_neighbor_rank = neighbor(mpi_topology, -1, 0, +1)
    ihikhi_neighbor_rank = neighbor(mpi_topology, +1, 0, +1)

    iloklo_halo_dom = domain[begin:(begin + nhalo - 1), :, begin:(begin + nhalo - 1)]
    ihiklo_halo_dom = domain[begin:(begin + nhalo - 1), :, (end - nhalo + 1):end]
    ilokhi_halo_dom = domain[(end - nhalo + 1):end, :, begin:(begin + nhalo - 1)]
    ihikhi_halo_dom = domain[(end - nhalo + 1):end, :, (end - nhalo + 1):end]

    iloklo_edge_dom = shift(shift(iloklo_halo_dom, iaxis, +nhalo), kaxis, +nhalo)
    ihiklo_edge_dom = shift(shift(ihiklo_halo_dom, iaxis, -nhalo), kaxis, +nhalo)
    ilokhi_edge_dom = shift(shift(ilokhi_halo_dom, iaxis, +nhalo), kaxis, -nhalo)
    ihikhi_edge_dom = shift(shift(ihikhi_halo_dom, iaxis, -nhalo), kaxis, -nhalo)

    if iloklo_neighbor_rank >= 0 && ihikhi_neighbor_rank >= 0
      ihikhi_edge = MPI.Buffer(view(A, ihikhi_edge_dom))
      iloklo_halo = MPI.Buffer(view(A, iloklo_halo_dom))

      MPI.Sendrecv!(
        ihikhi_edge, # send ihi-khi corner
        iloklo_halo, # to ilo-klo halo corner
        mpi_topology.comm;
        dest=ihikhi_neighbor_rank,    # to the ihi-khi rank
        source=iloklo_neighbor_rank,  # from the ilo-klo rank
      )

      iloklo_edge = MPI.Buffer(view(A, iloklo_edge_dom))
      ihikhi_halo = MPI.Buffer(view(A, ihikhi_halo_dom))

      MPI.Sendrecv!(
        iloklo_edge, # send ilo-klo corner
        ihikhi_halo, # to ihi-khi halo corner
        mpi_topology.comm;
        dest=iloklo_neighbor_rank,    # to the ilo-klo rank
        source=ihikhi_neighbor_rank,  # from the ihi-khi rank
      )
    end

    if ilokhi_neighbor_rank >= 0 && ihiklo_neighbor_rank >= 0
      ihiklo_edge = MPI.Buffer(view(A, ihiklo_edge_dom))
      ilokhi_halo = MPI.Buffer(view(A, ilokhi_halo_dom))

      MPI.Sendrecv!(
        ihiklo_edge, # send ihi-klo corner
        ilokhi_halo, # to ilo-khi halo 
        mpi_topology.comm;
        dest=ihiklo_neighbor_rank,    # to the ihi-klo rank
        source=ilokhi_neighbor_rank,  # from the ilo-khi rank
      )

      ilokhi_edge = MPI.Buffer(view(A, ilokhi_edge_dom))
      ihiklo_halo = MPI.Buffer(view(A, ihiklo_halo_dom))

      MPI.Sendrecv!(
        ilokhi_edge, # send ilo-khi corner
        ihiklo_halo, # to ihi-klo halo 
        mpi_topology.comm;
        dest=ilokhi_neighbor_rank,    # to the ilo-khi rank
        source=ihiklo_neighbor_rank,  # from the ihi-klo rank
      )
    end

    # ------

    jloklo_neighbor_rank = neighbor(mpi_topology, 0, -1, -1)
    jhiklo_neighbor_rank = neighbor(mpi_topology, 0, +1, -1)
    jlokhi_neighbor_rank = neighbor(mpi_topology, 0, -1, +1)
    jhikhi_neighbor_rank = neighbor(mpi_topology, 0, +1, +1)

    jloklo_halo_dom = domain[:, begin:(begin + nhalo - 1), begin:(begin + nhalo - 1)]
    jhiklo_halo_dom = domain[:, begin:(begin + nhalo - 1), (end - nhalo + 1):end]
    jlokhi_halo_dom = domain[:, (end - nhalo + 1):end, begin:(begin + nhalo - 1)]
    jhikhi_halo_dom = domain[:, (end - nhalo + 1):end, (end - nhalo + 1):end]

    jloklo_edge_dom = shift(jloklo_halo_dom, (jaxis, kaxis), (+nhalo, +nhalo))
    jhiklo_edge_dom = shift(jhiklo_halo_dom, (jaxis, kaxis), (-nhalo, +nhalo))
    jlokhi_edge_dom = shift(jlokhi_halo_dom, (jaxis, kaxis), (+nhalo, -nhalo))
    jhikhi_edge_dom = shift(jhikhi_halo_dom, (jaxis, kaxis), (-nhalo, -nhalo))

    if jloklo_neighbor_rank >= 0 && jhikhi_neighbor_rank >= 0
      jhikhi_edge = MPI.Buffer(view(A, jhikhi_edge_dom))
      jloklo_halo = MPI.Buffer(view(A, jloklo_halo_dom))

      MPI.Sendrecv!(
        jhikhi_edge, # send jhi-khi corner
        jloklo_halo, # to jlo-klo halo corner
        mpi_topology.comm;
        dest=jhikhi_neighbor_rank,    # to the jhi-khi rank
        source=jloklo_neighbor_rank,  # from the jlo-klo rank
      )

      jloklo_edge = MPI.Buffer(view(A, jloklo_edge_dom))
      jhikhi_halo = MPI.Buffer(view(A, jhikhi_halo_dom))

      MPI.Sendrecv!(
        jloklo_edge, # send jlo-klo corner
        jhikhi_halo, # to jhi-khi halo corner
        mpi_topology.comm;
        dest=jloklo_neighbor_rank,    # to the jlo-klo rank
        source=jhikhi_neighbor_rank,  # from the jhi-khi rank
      )
    end

    if jlokhi_neighbor_rank >= 0 && jhiklo_neighbor_rank >= 0
      jhiklo_edge = MPI.Buffer(view(A, jhiklo_edge_dom))
      jlokhi_halo = MPI.Buffer(view(A, jlokhi_halo_dom))

      MPI.Sendrecv!(
        jhiklo_edge, # send jhi-klo corner
        jlokhi_halo, # to jlo-khi halo 
        mpi_topology.comm;
        dest=jhiklo_neighbor_rank,    # to the jhi-klo rank
        source=jlokhi_neighbor_rank,  # from the jlo-khi rank
      )

      jlokhi_edge = MPI.Buffer(view(A, jlokhi_edge_dom))
      jhiklo_halo = MPI.Buffer(view(A, jhiklo_halo_dom))

      MPI.Sendrecv!(
        jlokhi_edge, # send jlo-khi corner
        jhiklo_halo, # to jhi-klo halo 
        mpi_topology.comm;
        dest=jlokhi_neighbor_rank,    # to the jlo-khi rank
        source=jhiklo_neighbor_rank,  # from the jhi-klo rank
      )
    end

    # ------

    ilojloklo_neighbor_rank = neighbor(mpi_topology, -1, -1, -1)
    ihijhikhi_neighbor_rank = neighbor(mpi_topology, +1, +1, +1)

    #! format: off
    ilojloklo_halo_dom = domain[begin:(begin + nhalo - 1), begin:(begin + nhalo - 1), begin:(begin + nhalo - 1)]
    ihijhikhi_halo_dom = domain[(end - nhalo + 1):end    , (end - nhalo + 1):end    , (end - nhalo + 1):end    ]

    ilojloklo_edge_dom = shift(ilojloklo_halo_dom, (iaxis, jaxis, kaxis), (+nhalo, +nhalo, +nhalo))
    ihijhikhi_edge_dom = shift(ihijhikhi_halo_dom, (iaxis, jaxis, kaxis), (-nhalo, -nhalo, -nhalo))
    #! format: on

    if ilojloklo_neighbor_rank >= 0 && ihijhikhi_neighbor_rank
      ilojloklo_edge = MPI.Buffer(view(A, ilojloklo_edge_dom))
      ihijhikhi_halo = MPI.Buffer(view(A, ihijhikhi_halo_dom))

      MPI.Sendrecv!(
        ilojloklo_edge, # send corner
        ihijhikhi_halo, # to halo 
        mpi_topology.comm;
        dest=ilojloklo_neighbor_rank,    # to 
        source=ihijhikhi_neighbor_rank,  # from 
      )

      ihijhikhi_edge = MPI.Buffer(view(A, ihijhikhi_edge_dom))
      ilojloklo_halo = MPI.Buffer(view(A, ilojloklo_halo_dom))

      MPI.Sendrecv!(
        ihijhikhi_edge, # send corner
        ilojloklo_halo, # to halo 
        mpi_topology.comm;
        dest=ihijhikhi_neighbor_rank,    # to
        source=ilojloklo_neighbor_rank,  # from
      )
    end

    ilojhiklo_neighbor_rank = neighbor(mpi_topology, -1, +1, -1)
    ihijlokhi_neighbor_rank = neighbor(mpi_topology, +1, -1, +1)

    #! format: off
    ilojhiklo_halo_dom = domain[begin:(begin + nhalo - 1), (end - nhalo + 1):end    , begin:(begin + nhalo - 1)]
    ihijlokhi_halo_dom = domain[(end - nhalo + 1):end    , begin:(begin + nhalo - 1), (end - nhalo + 1):end    ]
    ilojhiklo_edge_dom = shift(ilojhiklo_halo_dom, (iaxis,jaxis,kaxis), (+nhalo, -nhalo, +nhalo))
    ihijlokhi_edge_dom = shift(ihijlokhi_halo_dom, (iaxis,jaxis,kaxis), (-nhalo, +nhalo, -nhalo))
    #! format: on

    if ilojhiklo_neighbor_rank >= 0 && ihijlokhi_neighbor_rank
      ihijlokhi_edge = MPI.Buffer(view(A, ihijlokhi_edge_dom))
      ilojhiklo_halo = MPI.Buffer(view(A, ilojhiklo_halo_dom))

      MPI.Sendrecv!(
        ihijlokhi_edge, # send corner
        ilojhiklo_halo, # to halo 
        mpi_topology.comm;
        dest=ihijlokhi_neighbor_rank,    # to 
        source=ilojhiklo_neighbor_rank,  # from 
      )

      ilojhiklo_edge = MPI.Buffer(view(A, ilojhiklo_edge_dom))
      ihijlokhi_halo = MPI.Buffer(view(A, ihijlokhi_halo_dom))

      MPI.Sendrecv!(
        ilojhiklo_edge, # send corner
        ihijlokhi_halo, # to halo 
        mpi_topology.comm;
        dest=ilojhiklo_neighbor_rank,    # to
        source=ihijlokhi_neighbor_rank,  # from
      )
    end

    ihijloklo_neighbor_rank = neighbor(mpi_topology, +1, -1, -1)
    ilojhikhi_neighbor_rank = neighbor(mpi_topology, -1, +1, +1)

    #! format: off
    ihijloklo_halo_dom = domain[(end - nhalo + 1):end    , begin:(begin + nhalo - 1), begin:(begin + nhalo - 1)]
    ilojhikhi_halo_dom = domain[begin:(begin + nhalo - 1), (end - nhalo + 1):end    , (end - nhalo + 1):end    ]
    ihijloklo_edge_dom = shift(ihijloklo_halo_dom, (iaxis, jaxis, kaxis), (-nhalo, +nhalo, +nhalo))
    ilojhikhi_edge_dom = shift(ilojhikhi_halo_dom, (iaxis, jaxis, kaxis), (+nhalo, -nhalo, -nhalo))
    #! format: on

    if ihijloklo_neighbor_rank >= 0 && ilojhikhi_neighbor_rank
      ihijloklo_edge = MPI.Buffer(view(A, ihijloklo_edge_dom))
      ilojhikhi_halo = MPI.Buffer(view(A, ilojhikhi_halo_dom))

      MPI.Sendrecv!(
        ihijloklo_edge, # send corner
        ilojhikhi_halo, # to halo 
        mpi_topology.comm;
        dest=ihijloklo_neighbor_rank,    # to 
        source=ilojhikhi_neighbor_rank,  # from 
      )

      ilojhikhi_edge = MPI.Buffer(view(A, ilojhikhi_edge_dom))
      ihijloklo_halo = MPI.Buffer(view(A, ihijloklo_halo_dom))

      MPI.Sendrecv!(
        ilojhikhi_edge, # send corner
        ihijloklo_halo, # to halo 
        mpi_topology.comm;
        dest=ilojhikhi_neighbor_rank,    # to
        source=ihijloklo_neighbor_rank,  # from
      )
    end

    ihijhiklo_neighbor_rank = neighbor(mpi_topology, +1, +1, -1)
    ilojlokhi_neighbor_rank = neighbor(mpi_topology, -1, -1, +1)

    #! format: off
    ihijhiklo_halo_dom = domain[(end - nhalo + 1):end    , (end - nhalo + 1):end    , begin:(begin + nhalo - 1)]
    ilojlokhi_halo_dom = domain[begin:(begin + nhalo - 1), begin:(begin + nhalo - 1), (end - nhalo + 1):end    ]
    ihijhiklo_edge_dom = shift(ihijhiklo_halo_dom, (iaxis, jaxis, kaxis), (-nhalo, -nhalo, +nhalo))
    ilojlokhi_edge_dom = shift(ilojlokhi_halo_dom, (iaxis, jaxis, kaxis), (+nhalo, +nhalo, -nhalo))
    #! format: on

    if ihijhiklo_neighbor_rank >= 0 && ilojlokhi_neighbor_rank
      ihijhiklo_edge = MPI.Buffer(view(A, ihijhiklo_edge_dom))
      ilojlokhi_halo = MPI.Buffer(view(A, ilojlokhi_halo_dom))

      MPI.Sendrecv!(
        ihijhiklo_edge, # send corner
        ilojlokhi_halo, # to halo 
        mpi_topology.comm;
        dest=ihijhiklo_neighbor_rank,    # to 
        source=ilojlokhi_neighbor_rank,  # from 
      )

      ilojlokhi_edge = MPI.Buffer(view(A, ilojlokhi_edge_dom))
      ihijhiklo_halo = MPI.Buffer(view(A, ihijhiklo_halo_dom))
      MPI.Sendrecv!(
        ilojlokhi_edge, # send corner
        ihijhiklo_halo, # to halo 
        mpi_topology.comm;
        dest=ilojlokhi_neighbor_rank,    # to
        source=ihijhiklo_neighbor_rank,  # from
      )
    end
  end

  return nothing
end
