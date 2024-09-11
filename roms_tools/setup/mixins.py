from dataclasses import dataclass
from roms_tools.setup.grid import Grid
from roms_tools.setup.fill import lateral_fill
from roms_tools.setup.utils import (
    extrapolate_deepest_to_bottom,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    get_variable_metadata,
)
import xarray as xr
import numpy as np
import xesmf


@dataclass(frozen=True, kw_only=True)
class ROMSToolsMixins:
    """
    Represents a mixin tool for ROMS-Tools with capabilities shared by the various
    ROMS-Tools dataclasses.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.

    """

    grid: Grid

    def get_target_coords(self, use_coarse_grid=False):
        """
        Retrieves longitude and latitude coordinates from the grid, adjusting them based on orientation.

        Parameters
        ----------
        use_coarse_grid : bool, optional
            Use coarse grid data if True. Defaults to False.

        Returns
        -------
        dict
            Dictionary containing the longitude, latitude, and angle arrays, along with a boolean indicating
            if the grid straddles the meridian.
        """
        # Select grid variables based on whether the coarse grid is used
        if use_coarse_grid:
            lat, lon, angle = (
                self.grid.ds.lat_coarse,
                self.grid.ds.lon_coarse,
                self.grid.ds.angle_coarse,
            )
            lat_psi = self.grid.ds.get("lat_psi_coarse")
            lon_psi = self.grid.ds.get("lon_psi_coarse")
        else:
            lat, lon, angle = (
                self.grid.ds.lat_rho,
                self.grid.ds.lon_rho,
                self.grid.ds.angle,
            )
            lat_psi = self.grid.ds.get("lat_psi")
            lon_psi = self.grid.ds.get("lon_psi")

        # Operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        if lon_psi is not None:
            lon_psi = xr.where(lon_psi > 180, lon_psi - 360, lon_psi)

        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            if lon_psi is not None:
                lon_psi = xr.where(lon_psi < 0, lon_psi + 360, lon_psi)
            straddle = False

        target_coords = {
            "lat": lat,
            "lon": lon,
            "lat_psi": lat_psi,
            "lon_psi": lon_psi,
            "angle": angle,
            "straddle": straddle,
        }

        return target_coords

    def regrid_data(self, data, vars_2d, vars_3d, target_coords):

        """
        Interpolates data onto the desired grid and processes it for 2D and 3D variables.

        This method interpolates the specified 2D and 3D variables onto a new grid defined by the provided
        longitude and latitude coordinates. It handles both 2D and 3D data, performing extrapolation for 3D
        variables to fill values up to the bottom of the depth range.

        Parameters
        ----------
        data : DataContainer
            The container holding the variables to be interpolated. It must include attributes such as
            `dim_names` and `var_names`.
        vars_2d : list of str
            List of 2D variable names that should be interpolated.
        vars_3d : list of str
            List of 3D variable names that should be interpolated.
        lon : xarray.DataArray
            Longitude coordinates for interpolation.
        lat : xarray.DataArray
            Latitude coordinates for interpolation.

        Returns
        -------
        dict of str: xarray.DataArray
            A dictionary where keys are variable names and values are the interpolated DataArrays.

        Notes
        -----
        - 2D interpolation is performed using linear interpolation on the provided latitude and longitude coordinates.
        - For 3D variables, the method extrapolates the deepest values to the bottom of the depth range and interpolates
          using the specified depth coordinates.
        - The method assumes the presence of `dim_names` and `var_names` attributes in the `data` object.
        """

        # interpolate onto desired grid
        data_vars = {}

        ds_in, ds_out, corners_available = prepare_xesmf_regridding(data, target_coords)

        fill_dims = ["nlat", "nlon"]

        d_meta = get_variable_metadata()

        # 2d interpolation
        regridders = {}
        for var in vars_2d:

            # propagate ocean values into land areas
            if "time" in data.dim_names:
                mask = xr.where(
                    data.ds[data.var_names[var]]
                    .isel({data.dim_names["time"]: 0})
                    .isnull(),
                    0,
                    1,
                )
            else:
                mask = xr.where(data.ds[data.var_names[var]].isnull(), 0, 1)

            data.ds[data.var_names[var]] = lateral_fill(
                data.ds[data.var_names[var]].astype(np.float64).where(mask),
                1 - mask,
                dims=fill_dims,
            )

            # regrid
            method = d_meta[var]["preferred_regrid_method"]

            if method == "conservative" and not corners_available:
                method = "bilinear"

            if method not in regridders:
                regridders[method] = xesmf.Regridder(ds_in, ds_out, method=method)

            regridder = regridders[method]
            data_vars[var] = regridder(data.ds[data.var_names[var]])

        # 3d interpolation
        if vars_3d:

            ds_in["depth"] = data.ds[data.dim_names["depth"]]
            ds_out["layer_depth_rho"] = self.grid.ds["layer_depth_rho"]

            regridders = {}

            for var in vars_3d:
                # extrapolate deepest value all the way to bottom ("flooding")
                data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                    data.ds[data.var_names[var]], data.dim_names["depth"]
                )
                # propagate ocean values into land areas
                if "time" in data.dim_names:
                    mask = xr.where(
                        data.ds[data.var_names[var]]
                        .isel({data.dim_names["time"]: 0})
                        .isnull(),
                        0,
                        1,
                    )
                else:
                    mask = xr.where(data.ds[data.var_names[var]].isnull(), 0, 1)

                data.ds[data.var_names[var]] = lateral_fill(
                    data.ds[data.var_names[var]].astype(np.float64).where(mask),
                    1 - mask,
                    dims=fill_dims,
                )

                # regrid
                method = d_meta[var]["preferred_regrid_method"]

                if method == "conservative" and not corners_available:
                    method = "bilinear"

                if method not in regridders:
                    regridders[method] = xesmf.Regridder(ds_in, ds_out, method=method)

                regridder = regridders[method]
                data_vars[var] = regridder(data.ds[data.var_names[var]])

                # transpose to correct order (time, s_rho, eta_rho, xi_rho)
                data_vars[var] = data_vars[var].transpose(
                    "time", "s_rho", "eta_rho", "xi_rho"
                )

        return data_vars

    def process_velocities(self, data_vars, angle, uname, vname, interpolate=True):
        """
        Process and rotate velocity components to align with the grid orientation and optionally interpolate
        them to the appropriate grid points.

        This method performs the following steps:

        1. **Rotation**: Rotates the velocity components (e.g., `u`, `v`) to align with the grid orientation
           using the provided angle data.
        2. **Interpolation**: Optionally interpolates the rotated velocities from rho-points to u- and v-points
           of the grid.
        3. **Barotropic Velocity Calculation**: If the velocity components are 3D (with vertical coordinates),
           computes the barotropic (depth-averaged) velocities.

        Parameters
        ----------
        data_vars : dict of str: xarray.DataArray
            Dictionary containing the velocity components to be processed. The dictionary should include keys
            corresponding to the velocity component names (e.g., `uname`, `vname`).
        angle : xarray.DataArray
            DataArray containing the grid angle values used to rotate the velocity components to the correct
            orientation on the grid.
        uname : str
            The key corresponding to the zonal (east-west) velocity component in `data_vars`.
        vname : str
            The key corresponding to the meridional (north-south) velocity component in `data_vars`.
        interpolate : bool, optional
            If True, interpolates the rotated velocity components to the u- and v-points of the grid.
            Defaults to True.

        Returns
        -------
        dict of str: xarray.DataArray
            A dictionary of the processed velocity components. The returned dictionary includes the rotated and,
            if applicable, interpolated velocity components. If the input velocities are 3D (having a vertical
            dimension), the dictionary also includes the barotropic (depth-averaged) velocities (`ubar` and `vbar`).
        """

        regrid_method_u = data_vars[uname].attrs["regrid_method"]
        regrid_method_v = data_vars[vname].attrs["regrid_method"]

        # Rotate velocities to grid orientation
        u_rot = data_vars[uname] * np.cos(angle) + data_vars[vname] * np.sin(angle)
        v_rot = data_vars[vname] * np.cos(angle) - data_vars[uname] * np.sin(angle)

        # Interpolate to u- and v-points
        if interpolate:
            data_vars[uname] = interpolate_from_rho_to_u(u_rot)
            data_vars[vname] = interpolate_from_rho_to_v(v_rot)
        else:
            data_vars[uname] = u_rot
            data_vars[vname] = v_rot

        data_vars[uname] = data_vars[uname].assign_attrs(
            {"regrid_method": regrid_method_u}
        )
        data_vars[vname] = data_vars[vname].assign_attrs(
            {"regrid_method": regrid_method_v}
        )

        if "s_rho" in data_vars[uname].dims and "s_rho" in data_vars[vname].dims:
            # Compute barotropic velocity
            dz = -self.grid.ds["interface_depth_rho"].diff(dim="s_w")
            dz = dz.rename({"s_w": "s_rho"})
            dzu = interpolate_from_rho_to_u(dz)
            dzv = interpolate_from_rho_to_v(dz)

            data_vars["ubar"] = (
                (dzu * data_vars[uname]).sum(dim="s_rho") / dzu.sum(dim="s_rho")
            ).transpose("time", "eta_rho", "xi_u")
            data_vars["vbar"] = (
                (dzv * data_vars[vname]).sum(dim="s_rho") / dzv.sum(dim="s_rho")
            ).transpose("time", "eta_v", "xi_rho")

        return data_vars


def prepare_xesmf_regridding(data, target_coords):

    # output grid
    ds_out = xr.Dataset()
    ds_out["lon"] = target_coords["lon"]
    ds_out["lat"] = target_coords["lat"]

    # Assign bounding coordinates if corners are available
    corners_available = (
        target_coords["lon_psi"] is not None and target_coords["lat_psi"] is not None
    )
    if corners_available:
        ds_out["lon_b"] = target_coords["lon_psi"]
        ds_out["lat_b"] = target_coords["lat_psi"]

    # input grid
    ds_in = xr.Dataset()

    lon = data.ds[data.dim_names["longitude"]]
    lat = data.ds[data.dim_names["latitude"]]

    if corners_available:
        # compute corner points
        lon_b = 0.5 * (lon + lon.shift({data.dim_names["longitude"]: 1})).isel(
            {data.dim_names["longitude"]: slice(1, None)}
        ).rename({data.dim_names["longitude"]: "nlon_b"})
        lat_b = 0.5 * (lat + lat.shift({data.dim_names["latitude"]: 1})).isel(
            {data.dim_names["latitude"]: slice(1, None)}
        ).rename({data.dim_names["latitude"]: "nlat_b"})
        ds_in["lon_b"] = lon_b
        ds_in["lat_b"] = lat_b
        # cut off first and last cell center to make consistent with available corner points
        ds_in["lon"] = lon.isel({data.dim_names["longitude"]: slice(1, -1)}).rename(
            {data.dim_names["longitude"]: "nlon"}
        )
        ds_in["lat"] = lat.isel({data.dim_names["latitude"]: slice(1, -1)}).rename(
            {data.dim_names["latitude"]: "nlat"}
        )
        ds = data.ds.isel(
            {
                data.dim_names["longitude"]: slice(1, -1),
                data.dim_names["latitude"]: slice(1, -1),
            }
        )
        object.__setattr__(data, "ds", ds)
    else:
        ds_in["lon"] = lon.rename({data.dim_names["longitude"]: "nlon"})
        ds_in["lat"] = lat.rename({data.dim_names["latitude"]: "nlat"})

    ds = data.ds.rename(
        {data.dim_names["longitude"]: "nlon", data.dim_names["latitude"]: "nlat"}
    )
    object.__setattr__(data, "ds", ds)

    return ds_in, ds_out, corners_available
