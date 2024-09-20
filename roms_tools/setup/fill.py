import numpy as np
import xarray as xr
from numba import jit


def lateral_fill(var, dims=["latitude", "longitude"], method="sor", max_iter=10000):
    """
    Fills all NaN values in an xarray DataArray via a lateral fill, while leaving existing non-NaN values unchanged.

    Parameters
    ----------
    var : xarray.DataArray
        DataArray on which to fill NaNs. The fill is performed on the dimensions specified
        in `dims`.

    dims : list of str, optional, default=['latitude', 'longitude']
        Dimensions along which to perform the fill. The default is ['latitude', 'longitude'].

    method : str
        The fill method to use. Options are:
        - "sor": Successive Over-Relaxation (SOR) method.
        - "kara": Kara et al. extrapolation method.
    
    max_iter : int, optional, default=10000
        Maximum number of iterations to perform.

    Returns
    -------
    var_filled : xarray.DataArray
        DataArray with NaNs filled by iterative smoothing, except for the regions
        specified by `land_mask` where NaNs are preserved.

    """

    var_filled, iter_cnt = xr.apply_ufunc(
        _lateral_fill_np_array,
        var,
        input_core_dims=[dims],
        output_core_dims=[dims, []],
        output_dtypes=[var.dtype, np.int32],
        dask="parallelized",
        vectorize=True,
        kwargs={"method": method, "max_iter": max_iter},
    )

    return var_filled, iter_cnt


def _lateral_fill_np_array(var, method, max_iter=10000, fillvalue=0.0):
    """
    Fill NaN values in a NumPy array laterally using the specified method, while leaving
    existing non-NaN values unchanged.

    Parameters
    ----------
    var : np.ndarray
        A two-dimensional array in which to fill NaN values. Only NaNs in valid regions
        (determined by the `fillmask`) will be filled.
    method : str
        The fill method to use. Options are:
        - "sor": Successive Over-Relaxation (SOR) method.
        - "kara": Kara et al. extrapolation method.
    max_iter : int, optional, default=10000
        Maximum number of iterations to perform.
    fillvalue : float, optional
        The value to use if the entire array contains only NaNs. Default is 0.0.

    Returns
    -------
    np.ndarray
        The array with NaN values filled according to the specified method.

    Examples
    --------
    >>> import numpy as np
    >>> var = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> filled_var = _lateral_fill_np_array(var, method="sor")
    >>> print(filled_var)

    Notes
    -----
    - The "sor" method applies the Successive Over-Relaxation technique to fill missing values.
    - The "kara" method uses the approach described by Kara et al.
      (https://doi.org/10.1175/JPO2984.1) for extrapolating values over masked regions.
    """

    mask = np.isnan(var)  # fill all NaNs
    var = var.copy()

    if method == "sor":
        nlat, nlon = var.shape[-2:]
        var, iter_cnt = _iterative_fill_sor(nlat, nlon, var, mask, fillvalue, max_iter)

    elif method == "kara":

        var[np.isnan(var)] = 1e15
        var, iter_cnt = flood_kara_raw(var, 1 - mask.astype(int), max_iter)
        var[var==1e15] = np.nan

    return var, iter_cnt


@jit(nopython=True, parallel=True)
def _iterative_fill_sor(
    nlat, nlon, var, fillmask, fillvalue=0.0, max_iter=10000, tol=1.0e-4, rc=1.8
):
    """
    Perform an iterative land fill algorithm using the Successive Over-Relaxation (SOR)
    solution of the Laplace Equation.

    Parameters
    ----------
    nlat : int
        Number of latitude points in the array.

    nlon : int
        Number of longitude points in the array.

    var : numpy.array
        Two-dimensional array on which to fill NaNs.

    fillmask : numpy.array, boolean
        Mask indicating positions to be filled: `True` where data should be filled.

    fillvalue: float
        Value to use if the full field is NaNs. Default is 0.0.
    
    max_iter : int, optional, default=10000
        Maximum number of iterations to perform before giving up if the tolerance
        is not reached.

    tol : float, optional, default=1.0e-4
        Convergence criteria: stop filling when the value change is less than
        or equal to `tol * var`, i.e., `delta <= tol * np.abs(var[j, i])`.

    rc : float, optional, default=1.8
        Over-relaxation coefficient to use in the Successive Over-Relaxation (SOR)
        fill algorithm. Larger arrays (or extent of region to be filled if not global)
        typically converge faster with larger coefficients. For completely
        land-filling a 1-degree grid (360x180), a coefficient in the range 1.85-1.9
        is near optimal. Valid bounds are (1.0, 2.0).

    max_iter : int, optional, default=10000
        Maximum number of iterations to perform before giving up if the tolerance
        is not reached.

    Returns
    -------
    None
        The input array `var` is modified in-place with the NaN values filled.

    Notes
    -----
    This function performs the following steps:
    1. Computes a zonal mean to use as an initial guess for the fill.
    2. Replaces missing values in the input array with the computed zonal average.
    3. Iteratively fills the missing values using the SOR algorithm until the specified
       tolerance `tol` is reached or the maximum number of iterations `max_iter` is exceeded.

    Example
    -------
    >>> nlat, nlon = 180, 360
    >>> var = np.random.rand(nlat, nlon)
    >>> fillmask = np.isnan(var)
    >>> tol = 1.0e-4
    >>> rc = 1.8
    >>> max_iter = 10000
    >>> _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter)
    """

    # If field consists only of zeros, fill NaNs in with zeros and all done
    # Note: this will happen for shortwave downward radiation at night time
    if np.max(np.fabs(var)) == 0.0:
        var = np.zeros_like(var)
        return var, 0
    # If field consists only of NaNs, fill NaNs with fill value
    if np.isnan(var).all():
        var = fillvalue * np.ones_like(var)
        return var, 0

    # Compute a zonal mean to use as a first guess
    zoncnt = np.zeros(nlat)
    zonavg = np.zeros(nlat)
    for j in range(0, nlat):
        zoncnt[j] = np.sum(np.where(fillmask[j, :], 0, 1))
        zonavg[j] = np.sum(np.where(fillmask[j, :], 0, var[j, :]))
        if zoncnt[j] != 0:
            zonavg[j] = zonavg[j] / zoncnt[j]

    # Fill missing zonal averages for rows that are entirely land
    for j in range(0, nlat - 1):  # northward pass
        if zoncnt[j] > 0 and zoncnt[j + 1] == 0:
            zoncnt[j + 1] = 1
            zonavg[j + 1] = zonavg[j]
    for j in range(nlat - 1, 0, -1):  # southward pass
        if zoncnt[j] > 0 and zoncnt[j - 1] == 0:
            zoncnt[j - 1] = 1
            zonavg[j - 1] = zonavg[j]

    # Replace the input array missing values with zonal average as first guess
    for j in range(0, nlat):
        for i in range(0, nlon):
            if fillmask[j, i]:
                var[j, i] = zonavg[j]

    # Now do the iterative 2D fill
    res = np.zeros((nlat, nlon))  # work array hold residuals
    res_max = tol
    iter_cnt = 0
    while iter_cnt < max_iter and res_max >= tol:
        res[:] = 0.0  # reset the residual to zero for this iteration

        for j in range(1, nlat - 1):
            jm1 = j - 1
            jp1 = j + 1

            for i in range(1, nlon - 1):
                if fillmask[j, i]:
                    im1 = i - 1
                    ip1 = i + 1

                    # this is SOR
                    res[j, i] = (
                        var[j, ip1]
                        + var[j, im1]
                        + var[jm1, i]
                        + var[jp1, i]
                        - 4.0 * var[j, i]
                    )
                    var[j, i] = var[j, i] + rc * 0.25 * res[j, i]

        # do 1D smooth on top and bottom row if there is some valid data there in the input
        # otherwise leave it set to zonal average
        for j in [0, nlat - 1]:
            if zoncnt[j] > 1:

                for i in range(1, nlon - 1):
                    if fillmask[j, i]:
                        im1 = i - 1
                        ip1 = i + 1

                        res[j, i] = var[j, ip1] + var[j, im1] - 2.0 * var[j, i]
                        var[j, i] = var[j, i] + rc * 0.5 * res[j, i]

        # do 1D smooth in the vertical on left and right column
        for i in [0, nlon - 1]:

            for j in range(1, nlat - 1):
                if fillmask[j, i]:
                    jm1 = j - 1
                    jp1 = j + 1

                    res[j, i] = var[jp1, i] + var[jm1, i] - 2.0 * var[j, i]
                    var[j, i] = var[j, i] + rc * 0.5 * res[j, i]

        # four corners
        for j in [0, nlat - 1]:
            if j == 0:
                jp1 = j + 1
                jm1 = j
            elif j == nlat - 1:
                jp1 = j
                jm1 = j - 1

            for i in [0, nlon - 1]:
                if i == 0:
                    ip1 = i + 1
                    im1 = i
                elif i == nlon - 1:
                    ip1 = i
                    im1 = i - 1

                res[j, i] = (
                    var[j, ip1]
                    + var[j, im1]
                    + var[jm1, i]
                    + var[jp1, i]
                    - 4.0 * var[j, i]
                )
                var[j, i] = var[j, i] + rc * 0.25 * res[j, i]

        res_max = np.max(np.fabs(res)) / np.max(np.fabs(var))
        iter_cnt += 1
    print(iter_cnt)

    return var, iter_cnt


@jit(nopython=True, parallel=True)
def flood_kara_raw(field, mask, nmax=1000, fillvalue=0.0):
    """
    Extrapolate land values onto land using the Kara method.

    This function applies the method described by Kara et al.
    (https://doi.org/10.1175/JPO2984.1) to extrapolate values over land
    based on a binary land/sea mask.

    Parameters
    ----------
    field : np.ndarray
        The input field array to extrapolate, where NaN values represent
        areas to be filled.
    mask : np.ndarray
        A binary mask where 1 represents sea and 0 represents land.
    nmax : int, optional
        Maximum number of iterations for the extrapolation process,
        by default 1000.
    fillvalue : float, optional
        The value to use if the entire field consists of NaNs.
        Default is 0.0.

    Returns
    -------
    np.ndarray
        The field after extrapolation with NaN values replaced by
        extrapolated land values.

    Notes
    -----
    The Kara method uses an iterative extrapolation technique to fill
    in values over land areas. This method can be useful in oceanographic
    applications where missing data in land regions needs to be
    extrapolated from adjacent sea values.
    """

    # If field consists only of zeros, fill NaNs in with zeros and all done
    # Note: this will happen for shortwave downward radiation at night time
    if np.max(np.fabs(field)) == 0.0:
        field = np.zeros_like(field)
        return field, 0
    # If field consists only of NaNs, fill NaNs with fill value
    if np.isnan(field).all():
        field = fillvalue * np.ones_like(field)
        return field, 0

    ny, nx = field.shape
    nxy = nx * ny
    # create fields with halos
    ztmp = np.zeros((ny + 2, nx + 2))
    zmask = np.zeros((ny + 2, nx + 2))
    # init the values
    ztmp[1:-1, 1:-1] = field.copy()
    zmask[1:-1, 1:-1] = mask.copy()

    ztmp_new = ztmp.copy()
    zmask_new = zmask.copy()
    #
    nt = 0
    while (zmask[1:-1, 1:-1].sum() < nxy) and (nt < nmax):
        for jj in np.arange(1, ny + 1):
            for ji in np.arange(1, nx + 1):

                # compute once those indexes
                jjm1 = jj - 1
                jjp1 = jj + 1
                jim1 = ji - 1
                jip1 = ji + 1

                if zmask[jj, ji] == 0:
                    c6 = 1 * zmask[jjm1, jim1]
                    c7 = 2 * zmask[jjm1, ji]
                    c8 = 1 * zmask[jjm1, jip1]

                    c4 = 2 * zmask[jj, jim1]
                    c5 = 2 * zmask[jj, jip1]

                    c1 = 1 * zmask[jjp1, jim1]
                    c2 = 2 * zmask[jjp1, ji]
                    c3 = 1 * zmask[jjp1, jip1]

                    ctot = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

                    if ctot >= 3:
                        # compute the new value for this point
                        zval = (
                            c6 * ztmp[jjm1, jim1]
                            + c7 * ztmp[jjm1, ji]
                            + c8 * ztmp[jjm1, jip1]
                            + c4 * ztmp[jj, jim1]
                            + c5 * ztmp[jj, jip1]
                            + c1 * ztmp[jjp1, jim1]
                            + c2 * ztmp[jjp1, ji]
                            + c3 * ztmp[jjp1, jip1]
                        ) / ctot

                        # update value in field array
                        ztmp_new[jj, ji] = zval
                        # set the mask to sea
                        zmask_new[jj, ji] = 1
        nt += 1
        ztmp = ztmp_new.copy()
        zmask = zmask_new.copy()

    drowned = ztmp[1:-1, 1:-1]
    print(nt)

    return drowned, nt
