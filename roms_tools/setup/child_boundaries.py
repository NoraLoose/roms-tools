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


@dataclass(frozen=True, kw_only=True)
class ChildBoundaries:
    """
    Represents atmospheric forcing data for ocean modeling.

    Parameters
    ----------
    parent_grid : Grid
        Object representing the parent grid information.
    child_grid :
        Object representing the child grid information.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries of the child grid are to be forced (south, east, north, west). Default is all True.
    prefix : str
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
    prefix: str = "child"

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

        lon_parent = self.parent_grid.ds["lon_rho"]
        lat_parent = self.parent_grid.ds["lat_rho"]

        i_eta = np.arange(-0.5, len(lon_parent.eta_rho) + -0.5, 1)
        i_xi = np.arange(-0.5, len(lon_parent.xi_rho) + -0.5, 1)

        lon_parent = lon_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
            i_xi=("xi_rho", i_xi)
        )
        lat_parent = lat_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
            i_xi=("xi_rho", i_xi)
        )

        if self.parent_grid.straddle:
            lon_parent = xr.where(lon_parent > 180, lon_parent - 360, lon_parent)
        else:
            lon_parent = xr.where(lon_parent < 0, lon_parent + 360, lon_parent)

        child_grid_ds = self.child_grid.ds
        # add angles at u- and v-points
        child_grid_ds["angle_u"] = interpolate_from_rho_to_u(child_grid_ds["angle"])
        child_grid_ds["angle_v"] = interpolate_from_rho_to_v(child_grid_ds["angle"])

        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:
                for grid_location in ["rho", "u", "v"]:
                    if grid_location == "rho":
                        dim_names = {"latitude": "lat_rho", "longitude": "lon_rho"}
                        bdry_coords = bdry_coords_rho
                        suffix = "r"
                    elif grid_location == "u":
                        dim_names = {
                            "latitude": "lat_u",
                            "longitude": "lon_u",
                            "angle": "angle_u",
                        }
                        bdry_coords = bdry_coords_u
                        suffix = "u"
                    elif grid_location == "v":
                        dim_names = {
                            "latitude": "lat_v",
                            "longitude": "lon_v",
                            "angle": "angle_v",
                        }
                        bdry_coords = bdry_coords_v
                        suffix = "v"

                    lon_child = child_grid_ds[dim_names["longitude"]].isel(
                        **bdry_coords[direction]
                    )
                    lat_child = child_grid_ds[dim_names["latitude"]].isel(
                        **bdry_coords[direction]
                    )

                    if self.parent_grid.straddle:
                        lon_child = xr.where(
                            lon_child > 180, lon_child - 360, lon_child
                        )
                    else:
                        lon_child = xr.where(lon_child < 0, lon_child + 360, lon_child)

                    i_eta, i_xi = interpolate_indices(
                        lon_parent,
                        lat_parent,
                        lon_parent.i_eta,
                        lon_parent.i_xi,
                        lon_child,
                        lat_child,
                    )

                    if grid_location == "rho":
                        ds[f"{self.prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi], dim="two"
                        )  # dimension name "two" is suboptimal but inherited from matlab scripts
                    else:
                        angle_child = child_grid_ds[dim_names["angle"]].isel(
                            **bdry_coords[direction]
                        )
                        ds[f"{self.prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi, angle_child], dim="three"
                        )  # dimension name "three" is suboptimal but inherited from matlab scripts

        object.__setattr__(self, "ds", ds)


def interpolate_indices(lon_parent, lat_parent, i_parent, j_parent, lon, lat):
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

    return i, j
