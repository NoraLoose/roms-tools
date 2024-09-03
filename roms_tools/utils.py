from numbers import Integral

import numpy as np
import xarray as xr


def partition(
    ds: xr.Dataset, nx: int = 1, ny: int = 1
) -> tuple[list[int], list[xr.Dataset]]:
    """
    Partition a ROMS (Regional Ocean Modeling System) dataset into smaller spatial tiles.

    This function divides the input dataset into `nx` by `ny` tiles, where each tile
    represents a subdomain of the original dataset. The partitioning is performed along
    the spatial dimensions `eta_rho`, `xi_rho`, `eta_v`, `xi_u`, `eta_coarse`, and `xi_coarse`,
    depending on which dimensions are present in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input ROMS dataset that is to be partitioned.

    nx : int, optional
        The number of partitions along the `eta` (latitude) direction. Must be a positive integer. Default is 1.

    ny : int, optional
        The number of partitions along the `xi` (longitude) direction. Must be a positive integer. Default is 1.

    Returns
    -------
    tuple[list[int], list[xr.Dataset]]
        A tuple containing two elements:
        - A list of integers representing the file numbers associated with each partition.
        - A list of `xarray.Dataset` objects, each representing a partitioned subdomain of the original dataset.

    Raises
    ------
    ValueError
        If `nx` or `ny` is not a positive integer, or if the dataset cannot be evenly partitioned
        into the specified number of tiles.


    Example
    -------
    >>> partitioned_file_numbers, partitioned_datasets = partition(ds, nx=2, ny=3)
    >>> print(partitioned_file_numbers)
    [0, 1, 2, 3, 4, 5]
    >>> print([ds.sizes for ds in partitioned_datasets])
    [{'eta_rho': 50, 'xi_rho': 50}, {'eta_rho': 50, 'xi_rho': 50}, ...]

    This example partitions the dataset into 2 tiles along the `eta` direction and 3 tiles
    along the `xi` direction, resulting in a total of 6 partitions.
    """

    if not isinstance(nx, Integral) or nx < 1 or not isinstance(ny, Integral) or ny < 1:
        raise ValueError("nx and ny must be positive integers")

    partitionable_dims_maybe_present = [
        "eta_rho",
        "xi_rho",
        "eta_v",
        "xi_u",
        "eta_coarse",
        "xi_coarse",
    ]
    dims_to_partition = [d for d in partitionable_dims_maybe_present if d in ds.dims]

    # if eta is periodic there are no ghost cells along those dimensions
    if "eta_v" in ds.sizes and ds.sizes["eta_rho"] == ds.sizes["eta_v"]:
        # TODO how are we supposed to know if eta is periodic if eta_v doesn't appear? partit.F doesn't say...
        n_eta_ghost_cells = 0
    else:
        n_eta_ghost_cells = 1

    # if xi is periodic there are no ghost cells along those dimensions
    if "xi_u" in ds.sizes and ds.sizes["xi_rho"] == ds.sizes["xi_u"]:
        n_xi_ghost_cells = 0
    else:
        n_xi_ghost_cells = 1

    def integer_division_or_raise(a: int, b: int, dimension: str) -> int:
        """
        Perform integer division and ensure that the division is exact.

        Parameters
        ----------
        a : int
            The numerator for the division.
        b : int
            The denominator for the division.
        dimension : str
            The name of the dimension being partitioned, used for error reporting.

        Returns
        -------
        int
            The result of the integer division.

        Raises
        ------
        ValueError
            If the division is not exact, indicating that the domain cannot be evenly divided
            along the specified dimension.
        """
        remainder = a % b
        if remainder == 0:
            return a // b
        else:
            raise ValueError(
                f"Dimension '{dimension}' of size {a} cannot be evenly divided into {b} partitions."
            )

    if "eta_rho" in dims_to_partition:
        eta_rho_domain_size = integer_division_or_raise(
            ds.sizes["eta_rho"] - 2 * n_eta_ghost_cells, nx, "eta_rho"
        )
    if "xi_rho" in dims_to_partition:
        xi_rho_domain_size = integer_division_or_raise(
            ds.sizes["xi_rho"] - 2 * n_xi_ghost_cells, ny, "xi_rho"
        )

    if "eta_v" in dims_to_partition:
        eta_v_domain_size = integer_division_or_raise(
            ds.sizes["eta_v"] - 1 * n_eta_ghost_cells, nx, "eta_v"
        )
    if "xi_u" in dims_to_partition:
        xi_u_domain_size = integer_division_or_raise(
            ds.sizes["xi_u"] - 1 * n_xi_ghost_cells, ny, "xi_u"
        )

    if "eta_coarse" in dims_to_partition:
        eta_coarse_domain_size = integer_division_or_raise(
            ds.sizes["eta_coarse"] - 2 * n_eta_ghost_cells, nx, "eta_coarse"
        )
    if "xi_coarse" in dims_to_partition:
        xi_coarse_domain_size = integer_division_or_raise(
            ds.sizes["xi_coarse"] - 2 * n_xi_ghost_cells, ny, "xi_coarse"
        )

    # unpartitioned dimensions should have sizes unchanged
    partitioned_sizes = {
        dim: [size] for dim, size in ds.sizes.items() if dim in dims_to_partition
    }

    # TODO refactor to use two functions for odd- and even-length dimensions
    if "eta_v" in dims_to_partition:
        partitioned_sizes["eta_v"] = [eta_v_domain_size] * (nx - 1) + [
            eta_v_domain_size + n_eta_ghost_cells
        ]
    if "xi_u" in dims_to_partition:
        partitioned_sizes["xi_u"] = [xi_u_domain_size] * (ny - 1) + [
            xi_u_domain_size + n_xi_ghost_cells
        ]

    if nx > 1:
        partitioned_sizes["eta_rho"] = (
            [eta_rho_domain_size + n_eta_ghost_cells]
            + [eta_rho_domain_size] * (nx - 2)
            + [eta_rho_domain_size + n_eta_ghost_cells]
        )

        if "eta_coarse" in dims_to_partition:
            partitioned_sizes["eta_coarse"] = (
                [eta_coarse_domain_size + n_eta_ghost_cells]
                + [eta_coarse_domain_size] * (nx - 2)
                + [eta_coarse_domain_size + n_eta_ghost_cells]
            )

    if ny > 1:
        partitioned_sizes["xi_rho"] = (
            [xi_rho_domain_size + n_xi_ghost_cells]
            + [xi_rho_domain_size] * (ny - 2)
            + [xi_rho_domain_size + n_xi_ghost_cells]
        )

        if "xi_coarse" in dims_to_partition:
            partitioned_sizes["xi_coarse"] = (
                [xi_coarse_domain_size + n_xi_ghost_cells]
                + [xi_coarse_domain_size] * (ny - 2)
                + [xi_coarse_domain_size + n_xi_ghost_cells]
            )

    def cumsum(pmf):
        """Implementation of cumsum which ensures the result starts with zero"""
        cdf = np.empty(len(pmf) + 1, dtype=int)
        cdf[0] = 0
        np.cumsum(pmf, out=cdf[1:])
        return cdf

    file_numbers = []
    partitioned_datasets = []
    for j in range(ny):
        for i in range(nx):
            file_number = i + (j * ny)
            file_numbers.append(file_number)

            indexers = {}

            if "eta_rho" in dims_to_partition:
                eta_rho_partition_indices = cumsum(partitioned_sizes["eta_rho"])
                indexers["eta_rho"] = slice(
                    int(eta_rho_partition_indices[i]),
                    int(eta_rho_partition_indices[i + 1]),
                )
            if "xi_rho" in dims_to_partition:
                xi_rho_partition_indices = cumsum(partitioned_sizes["xi_rho"])
                indexers["xi_rho"] = slice(
                    int(xi_rho_partition_indices[j]),
                    int(xi_rho_partition_indices[j + 1]),
                )

            if "eta_v" in dims_to_partition:
                eta_v_partition_indices = cumsum(partitioned_sizes["eta_v"])
                indexers["eta_v"] = slice(
                    int(eta_v_partition_indices[i]),
                    int(eta_v_partition_indices[i + 1]),
                )
            if "xi_u" in dims_to_partition:
                xi_u_partition_indices = cumsum(partitioned_sizes["xi_u"])
                indexers["xi_u"] = slice(
                    int(xi_u_partition_indices[j]), int(xi_u_partition_indices[j + 1])
                )

            if "eta_coarse" in dims_to_partition:
                eta_coarse_partition_indices = cumsum(partitioned_sizes["eta_coarse"])
                indexers["eta_coarse"] = slice(
                    int(eta_coarse_partition_indices[i]),
                    int(eta_coarse_partition_indices[i + 1]),
                )

            if "xi_coarse" in dims_to_partition:
                xi_coarse_partition_indices = cumsum(partitioned_sizes["xi_coarse"])
                indexers["xi_coarse"] = slice(
                    int(xi_coarse_partition_indices[j]),
                    int(xi_coarse_partition_indices[j + 1]),
                )

            partitioned_ds = ds.isel(**indexers)

            partitioned_datasets.append(partitioned_ds)

    return file_numbers, partitioned_datasets
