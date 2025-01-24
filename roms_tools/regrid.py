import xarray as xr


class LateralRegrid:
    """Handles lateral regridding of data onto a new spatial grid.

    Parameters
    ----------
    source_grid : dict
        Dictionary containing the source grid information. It should have:
        - 'dim_names': A dictionary specifying names for the latitude and longitude dimensions
                       (e.g., {"latitude": "lat", "longitude": "lon"}).
        - 'coords': A dictionary of xarray.DataArrays for the source latitude and longitude,
                    typically with keys matching 'dim_names'.
    target_coords : dict
        Dictionary containing 'lon' and 'lat' as xarray.DataArrays representing
        the longitude and latitude values of the target grid.
    use_xesmf : bool, optional
        If True, use xESMF for regridding. If False, use xarray's interpolation.

    Attributes
    ----------
    use_xesmf : bool
        Indicates whether to use xESMF for regridding.
    coords : dict
        Maps source dimension names to the corresponding latitude and longitude
        DataArrays for the target grid (only used if `use_xesmf=False`).
    regridder : xesmf.Regridder or None
        xESMF regridder object (only used if `use_xesmf=True`).
    """

    def __init__(self, source_grid, target_coords, use_xesmf=False):
        self.use_xesmf = use_xesmf

        if self.use_xesmf:

            # Prepare source and target grids for xESMF
            self.regridder = self._initialize_xesmf_regridder(
                source_grid, target_coords
            )
        else:
            # Prepare target grid coordinates for xarray interpolation
            dim_names = source_grid["dim_names"]
            self.coords = {
                dim_names["latitude"]: target_coords["lat"],
                dim_names["longitude"]: target_coords["lon"],
            }

    def _initialize_xesmf_regridder(self, source_grid, target_coords):
        """Initializes an xESMF regridder."""
        import xesmf

        dim_names = source_grid["dim_names"]
        source_ds = xr.Dataset()
        source_ds["lon"] = source_grid["coords"][dim_names["longitude"]].rename(
            {dim_names["longitude"]: "nlon"}
        )
        source_ds["lat"] = source_grid["coords"][dim_names["latitude"]].rename(
            {dim_names["latitude"]: "nlat"}
        )

        target_ds = xr.Dataset()
        target_ds["lon"] = target_coords["lon"]
        target_ds["lat"] = target_coords["lat"]

        return xesmf.Regridder(
            source_ds, target_ds, method="bilinear", reuse_weights=True
        )

    def apply(self, da):
        """Regrids the input variable to the target grid.

        Parameters
        ----------
        da : xarray.DataArray
            The input data to regrid. This should have coordinates matching the source grid.

        Returns
        -------
        xarray.DataArray
            The regridded data aligned to the target grid.
        """
        if self.use_xesmf:
            regridded = self.regridder(da)
        else:
            method = "linear"
            # Regrid using xarray's built-in interpolation
            regridded = da.interp(self.coords, method=method).drop_vars(
                list(self.coords.keys())
            )
        return regridded


class VerticalRegrid:
    """Interpolates data onto new vertical (depth) coordinates.

    Parameters
    ----------
    target_depth_coords : xarray.DataArray
        Depth coordinates for the target grid.
    source_depth_coords : xarray.DataArray
        Depth coordinates for the source grid.
    """

    def __init__(self, target_depth_coords, source_depth_coords):
        """Initialize regridding factors for interpolation.

        Parameters
        ----------
        target_depth_coords : xarray.DataArray
            Depth coordinates for the target grid.
        source_depth_coords : xarray.DataArray
            Depth coordinates for the source grid.

        Attributes
        ----------
        coeff : xarray.Dataset
            Dataset containing:
            - `is_below` : Boolean mask for depths just below target.
            - `is_above` : Boolean mask for depths just above target.
            - `upper_mask`, `lower_mask` : Masks for valid interpolation bounds.
            - `factor` : Weight for blending values between levels.
        """

        self.depth_dim = source_depth_coords.dims[0]
        source_depth = source_depth_coords
        dims = {"dim": self.depth_dim}

        dlev = source_depth - target_depth_coords
        is_below = dlev == dlev.where(dlev >= 0).min(**dims)
        is_above = dlev == dlev.where(dlev <= 0).max(**dims)
        p_below = dlev.where(is_below).sum(**dims)
        p_above = -dlev.where(is_above).sum(**dims)
        denominator = p_below + p_above
        denominator = denominator.where(denominator > 1e-6, 1e-6)
        factor = p_below / denominator

        upper_mask = is_above.sum(**dims) > 0
        lower_mask = is_below.sum(**dims) > 0

        self.coeff = xr.Dataset(
            {
                "is_below": is_below,
                "is_above": is_above,
                "upper_mask": upper_mask,
                "lower_mask": lower_mask,
                "factor": factor,
            }
        )

    def apply(self, var, fill_nans=True):
        """Interpolates the variable onto the new depth grid using precomputed
        coefficients for linear interpolation between layers.

        Parameters
        ----------
        var : xarray.DataArray
            The input data to be regridded along the depth dimension. This should be
            an array with the same depth coordinates as the original grid.
        fill_nans : bool, optional
            Whether to fill NaN values in the regridded data. If True (default),
            forward-fill and backward-fill are applied along the 's_rho' dimension to
            ensure there are no NaNs after interpolation.

        Returns
        -------
        xarray.DataArray
            The regridded data array, interpolated onto the new depth grid. NaN values
            are replaced if `fill_nans=True`, with extrapolation allowed at the surface
            and bottom layers to minimize gaps.
        """

        dims = {"dim": self.depth_dim}

        var_below = var.where(self.coeff["is_below"]).sum(**dims)
        var_above = var.where(self.coeff["is_above"]).sum(**dims)

        result = var_below + (var_above - var_below) * self.coeff["factor"]
        if fill_nans:
            result = result.where(self.coeff["upper_mask"], var.isel({dims["dim"]: 0}))
            result = result.where(self.coeff["lower_mask"], var.isel({dims["dim"]: -1}))
        else:
            result = result.where(self.coeff["upper_mask"]).where(
                self.coeff["lower_mask"]
            )

        return result
