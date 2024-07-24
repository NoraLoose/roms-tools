import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from dataclasses import dataclass, field
from typing import Dict
from roms_tools.setup.grid import Grid
from roms_tools.setup.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
from roms_tools.setup.plot import _plot_nesting
import warnings


@dataclass(frozen=True, kw_only=True)
class NestBoundaries:
    """
    Represents relation between parent and child grid for nested ROMS simulations.

    Parameters
    ----------
    parent_grid : Grid
        Object representing the parent grid information.
    child_grid :
        Object representing the child grid information.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries of the child grid are to be forced (south, east, north, west). Default is all True.
    child_prefix : str
    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the index information for the child grid.
    """

    parent_grid: Grid
    child_grid: Grid
    boundaries: Dict[str, bool] = field(
        default_factory=lambda: {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
    )
    child_prefix: str = "child"

    def __post_init__(self):

        # Boundary coordinates for rho-points
        bdry_coords_rho = {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        }

        # Boundary coordinates for u-points
        bdry_coords_u = {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        }

        # Boundary coordinates for v-points
        bdry_coords_v = {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        }

        #lon_parent = self.parent_grid.ds["lon_rho"]
        #lat_parent = self.parent_grid.ds["lat_rho"]
        parent_grid_ds = self.parent_grid_ds
        child_grid_ds = self.child_grid.ds

        #i_eta = np.arange(-0.5, len(lon_parent.eta_rho) + -0.5, 1)
        #i_xi = np.arange(-0.5, len(lon_parent.xi_rho) + -0.5, 1)
        i_eta = np.arange(-0.5, len(parent_grid_ds.eta_rho) + -0.5, 1)
        i_xi = np.arange(-0.5, len(parent_grid_ds.xi_rho) + -0.5, 1)

        #lon_parent = lon_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
        #    i_xi=("xi_rho", i_xi)
        #)
        #lat_parent = lat_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
        #    i_xi=("xi_rho", i_xi)
        #)
        parent_grid_ds = parent_grid_ds.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
            i_xi=("xi_rho", i_xi)
                    if self.parent_grid.straddle:
                        lon_child = xr.where(
                            lon_child > 180, lon_child - 360, lon_child
                        )
                    else:
                        lon_child = xr.where(lon_child < 0, lon_child + 360, lon_child)

        if self.parent_grid.straddle:
            #lon_parent = xr.where(lon_parent > 180, lon_parent - 360, lon_parent)
            for grid_ds in [parent_grid_ds, child_grid_ds]:
                grid_ds["lon_rho"] = xr.where(grid_ds["lon_rho"] > 180, grid_ds["lon_rho"] - 360, grid_ds["lon_rho"])
        else:
            #lon_parent = xr.where(lon_parent < 0, lon_parent + 360, lon_parent)
            for grid_ds in [parent_grid_ds, child_grid_ds]:
                grid_ds["lon_rho"] = xr.where(grid_ds["lon_rho"] < 0, grid_ds["lon_rho"] + 360, grid_ds["lon_rho"])

        # add angles at u- and v-points
        child_grid_ds["angle_u"] = interpolate_from_rho_to_u(child_grid_ds["angle"])
        child_grid_ds["angle_v"] = interpolate_from_rho_to_v(child_grid_ds["angle"])

        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:
                for grid_location in ["rho", "u", "v"]:
                    if grid_location == "rho":
                        names = {"latitude": "lat_rho", "longitude": "lon_rho", "mask": "mask_rho"}
                        bdry_coords = bdry_coords_rho
                        suffix = "r"
                    elif grid_location == "u":
                        names = {
                            "latitude": "lat_u",
                            "longitude": "lon_u",
                            "mask": "mask_u",
                            "angle": "angle_u",
                        }
                        bdry_coords = bdry_coords_u
                        suffix = "u"
                    elif grid_location == "v":
                        names = {
                            "latitude": "lat_v",
                            "longitude": "lon_v",
                            "mask": "mask_v",
                            "angle": "angle_v",
                        }
                        bdry_coords = bdry_coords_v
                        suffix = "v"

                    lon_child = child_grid_ds[names["longitude"]].isel(
                        **bdry_coords[direction]
                    )
                    lat_child = child_grid_ds[names["latitude"]].isel(
                        **bdry_coords[direction]
                    )
                    mask_child = child_grid_ds[names["mask"]].isel(
                        **bdry_coords[direction]
                    )
                    #if self.parent_grid.straddle:
                    #    lon_child = xr.where(
                    #        lon_child > 180, lon_child - 360, lon_child
                    #    )
                    #else:
                    #    lon_child = xr.where(lon_child < 0, lon_child + 360, lon_child)
                    
                    # Crop parent grid to minimial size to avoid aliased interpolated indices

                    i_eta, i_xi = interpolate_indices(
                        lon_parent,
                        lat_parent,
                        lon_parent.i_eta,
                        lon_parent.i_xi,
                        lon_child,
                        lat_child,
                        mask_child
                    )

                    if grid_location == "rho":
                        ds[f"{self.child_prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi], dim="two"
                        )  # dimension name "two" is suboptimal but inherited from matlab scripts
                    else:
                        angle_child = child_grid_ds[names["angle"]].isel(
                            **bdry_coords[direction]
                        )
                        ds[f"{self.child_prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi, angle_child], dim="three"
                        )  # dimension name "three" is suboptimal but inherited from matlab scripts

        # Rename dimensions
        dims_to_rename = {dim: f"{self.child_prefix}_{dim}" for dim in ds.dims if dim not in ["two", "three"]}
        ds = ds.rename(dims_to_rename)

        object.__setattr__(self, "ds", ds)

    def plot(self) -> None:
        """
        Plot the parent and child grids in a single figure.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        """

        _plot_nesting(self.parent_grid.ds, self.child_grid.ds, self.parent_grid.straddle)


def refine_region(parent_ds, latitude_range, longitude_range, iterations=5):
    """
    Refine the region of the grid to match boundary conditions.

    Parameters
    ----------
    lons : xarray.DataArray
        Longitudes of the grid.
    lats : xarray.DataArray
        Latitudes of the grid.
    ips : xarray.DataArray
        i-indices of the grid.
    jps : xarray.DataArray
        j-indices of the grid.
    lonbc_mn : float
        Minimum longitude boundary condition.
    lonbc_mx : float
        Maximum longitude boundary condition.
    latbc_mn : float
        Minimum latitude boundary condition.
    latbc_mx : float
        Maximum latitude boundary condition.
    iterations : int, optional
        Number of iterations to refine the region. Default is 5.

    Returns
    -------
    lons : xarray.DataArray
        Refined longitudes.
    lats : xarray.DataArray
        Refined latitudes.
    ips : xarray.DataArray
        Refined i-indices.
    jps : xarray.DataArray
        Refined j-indices.
    """
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    for _ in range(iterations):
        nxs, nys = lons.shape

        parent_lon_min = parent_ds["lon_rho"].min()
        lon_mx = lons.max(dim='xi')
        lat_mn = lats.min(dim='eta')
        lat_mx = lats.max(dim='eta')

        i0 = (lon_mx < lonbc_mn).argmin().item() if (lon_mx < lonbc_mn).any() else 0
        i1 = (lon_mn > lonbc_mx).argmax().item() if (lon_mn > lonbc_mx).any() else nxs - 1
        j0 = (lat_mx < latbc_mn).argmin().item() if (lat_mx < latbc_mn).any() else 0
        j1 = (lat_mn > latbc_mx).argmax().item() if (lat_mn > latbc_mx).any() else nys - 1

        lons = lons.isel(xi=slice(i0, i1 + 1), eta=slice(j0, j1 + 1))
        lats = lats.isel(xi=slice(i0, i1 + 1), eta=slice(j0, j1 + 1))
        ips = ips.isel(xi=slice(i0, i1 + 1), eta=slice(j0, j1 + 1))
        jps = jps.isel(xi=slice(i0, i1 + 1), eta=slice(j0, j1 + 1))

    return lons, lats, ips, jps


def interpolate_indices(lon_parent, lat_parent, i_parent, j_parent, lon, lat, mask):
    """
    Interpolate the parent indices to the child grid.

    Parameters
    ----------
    lon_parent : xarray.DataArray
        Longitudes of the parent grid.
    lat_parent : xarray.DataArray
        Latitudes of the parent grid.
    i_parent : xarray.DataArray
        i-indices of the parent grid.
    j_parent : xarray.DataArray
        j-indices of the parent grid.
    lon : xarray.DataArray
        Longitudes of the child grid where interpolation is desired.
    lat : xarray.DataArray
        Latitudes of the child grid where interpolation is desired.
    mask: xarray.DataArray
        Mask for the child grid.
    Returns
    -------
    i : xarray.DataArray
        Interpolated i-indices for the child grid.
    j : xarray.DataArray
        Interpolated j-indices for the child grid.
    """

    # Create meshgrid
    i_parent, j_parent = np.meshgrid(i_parent.values, j_parent.values)

    # Flatten the input coordinates and indices for griddata
    points = np.column_stack((lon_parent.values.ravel(), lat_parent.values.ravel()))
    i_parent_flat = i_parent.ravel()
    j_parent_flat = j_parent.ravel()

    # Interpolate the i and j indices
    i = griddata(points, i_parent_flat, (lon.values, lat.values), method="linear")
    j = griddata(points, j_parent_flat, (lon.values, lat.values), method="linear")

    i = xr.DataArray(i, dims=lon.dims)
    j = xr.DataArray(j, dims=lon.dims)

    # Check unmasked i- and j-indices
    i_chk = i[mask == 1]
    j_chk = j[mask == 1]

    # Check for NaN values
    if np.sum(np.isnan(i_chk)) > 0 or np.sum(np.isnan(j_chk)):
        raise ValueError('Some unmasked points are outside the grid. Please choose either a bigger parent grid or a smaller child grid.')
    
    nxp, nyp = lon_parent.shape
    # Check whether indices are close to border of parent grid
    if len(i_chk) > 0:
        if np.min(i_chk) < 0 or np.max(i_chk) > nxp - 2:
            warnings.warn('Some points are borderline.')
    if len(j_chk) > 0:
        if np.min(j_chk) < 0 or np.max(j_chk) > nyp - 2:
            warnings.warn('Some points are borderline.')

    return i, j
