import pytest
import numpy as np
import numpy.testing as npt
from scipy.ndimage import label
from roms_tools import Grid


class TestCreateGrid:
    def test_simple_regression(self):
        grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0)

        expected_lat = np.array(
            [
                [-8.99249453e-01, -8.99249453e-01, -8.99249453e-01],
                [0.0, 0.0, 0.0],
                [ 8.99249453e-01,  8.99249453e-01,  8.99249453e-01],
            ]
        )
        expected_lon = np.array(
            [
                [339.10072286, 340.        , 340.89927714],
                [339.10072286, 340.        , 340.89927714],
                [339.10072286, 340.        , 340.89927714],
            ]
        )

        # TODO: adapt tolerances according to order of magnitude of respective fields
        npt.assert_allclose(grid.ds["lat_rho"], expected_lat, atol=1e-8)
        npt.assert_allclose(grid.ds["lon_rho"], expected_lon, atol=1e-8)

    def test_raise_if_domain_too_large(self):
        with pytest.raises(ValueError, match="Domain size has to be smaller"):
            Grid(nx=3, ny=3, size_x=30000, size_y=30000, center_lon=0, center_lat=51.5)

        # test grid with reasonable domain size
        grid = Grid(
            nx=3,
            ny=3,
            size_x=1800,
            size_y=2400,
            center_lon=-21,
            center_lat=61,
            rot=20,
        )
        assert isinstance(grid, Grid)


class TestGridFromFile:
    def test_equal_to_from_init(self):
        ...

    def test_roundtrip(self):
        """Test that creating a grid, saving it to file, and re-opening it is the same as just creating it."""
        ...


class TestTopography:
    def test_enclosed_regions(self):
        """Test that there are only two connected regions, one dry and one wet."""

        grid = Grid(
            nx=100,
            ny=100,
            size_x=1800,
            size_y=2400,
            center_lon=30,
            center_lat=61,
            rot=20,
        )

        reg, nreg = label(grid.ds.mask_rho_filled)
        npt.assert_equal(nreg, 2)
