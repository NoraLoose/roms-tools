import pytest
from datetime import datetime
import numpy as np
import xarray as xr
from roms_tools.setup.datasets import Dataset, GLORYSDataset, ERA5Correction
from roms_tools.setup.download import download_test_data
import tempfile
import os
from pathlib import Path


@pytest.fixture
def global_dataset():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    depth = np.linspace(0, 2000, 10)
    time = [
        np.datetime64("2022-01-01T00:00:00"),
        np.datetime64("2022-02-01T00:00:00"),
        np.datetime64("2022-03-01T00:00:00"),
        np.datetime64("2022-04-01T00:00:00"),
    ]
    data = np.random.rand(4, 10, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "depth", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "depth": (["depth"], depth),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def global_dataset_with_noon_times():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    time = [
        np.datetime64("2022-01-01T12:00:00"),
        np.datetime64("2022-02-01T12:00:00"),
        np.datetime64("2022-03-01T12:00:00"),
        np.datetime64("2022-04-01T12:00:00"),
    ]
    data = np.random.rand(4, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def global_dataset_with_multiple_times_per_day():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    time = [
        np.datetime64("2022-01-01T00:00:00"),
        np.datetime64("2022-01-01T12:00:00"),
        np.datetime64("2022-02-01T00:00:00"),
        np.datetime64("2022-02-01T12:00:00"),
        np.datetime64("2022-03-01T00:00:00"),
        np.datetime64("2022-03-01T12:00:00"),
        np.datetime64("2022-04-01T00:00:00"),
        np.datetime64("2022-04-01T12:00:00"),
    ]
    data = np.random.rand(8, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def non_global_dataset():
    lon = np.linspace(0, 180, 181)
    lat = np.linspace(-90, 90, 180)
    data = np.random.rand(180, 181)
    ds = xr.Dataset(
        {"var": (["latitude", "longitude"], data)},
        coords={"latitude": (["latitude"], lat), "longitude": (["longitude"], lon)},
    )
    return ds


@pytest.mark.parametrize(
    "data_fixture, expected_time_values",
    [
        (
            "global_dataset",
            [
                np.datetime64("2022-02-01T00:00:00"),
                np.datetime64("2022-03-01T00:00:00"),
            ],
        ),
        (
            "global_dataset_with_noon_times",
            [
                np.datetime64("2022-01-01T12:00:00"),
                np.datetime64("2022-02-01T12:00:00"),
                np.datetime64("2022-03-01T12:00:00"),
            ],
        ),
        (
            "global_dataset_with_multiple_times_per_day",
            [
                np.datetime64("2022-02-01T00:00:00"),
                np.datetime64("2022-02-01T12:00:00"),
                np.datetime64("2022-03-01T00:00:00"),
            ],
        ),
    ],
)
def test_select_times(data_fixture, expected_time_values, request, use_dask):
    """
    Test selecting times with different datasets.
    """
    start_time = datetime(2022, 2, 1)
    end_time = datetime(2022, 3, 1)

    # Get the fixture dynamically based on the parameter
    dataset = request.getfixturevalue(data_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )

        assert dataset.ds is not None
        assert len(dataset.ds.time) == len(expected_time_values)
        for expected_time in expected_time_values:
            assert expected_time in dataset.ds.time.values
    finally:
        os.remove(filepath)


@pytest.mark.parametrize(
    "data_fixture, expected_time_values",
    [
        ("global_dataset", [np.datetime64("2022-02-01T00:00:00")]),
        ("global_dataset_with_noon_times", [np.datetime64("2022-02-01T12:00:00")]),
    ],
)
def test_select_times_no_end_time(
    data_fixture, expected_time_values, request, use_dask
):
    """
    Test selecting times with only start_time specified.
    """
    start_time = datetime(2022, 2, 1)

    # Get the fixture dynamically based on the parameter
    dataset = request.getfixturevalue(data_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            use_dask=use_dask,
        )

        assert dataset.ds is not None
        assert len(dataset.ds.time) == len(expected_time_values)
        for expected_time in expected_time_values:
            assert expected_time in dataset.ds.time.values
    finally:
        os.remove(filepath)


def test_multiple_matching_times(global_dataset_with_multiple_times_per_day, use_dask):
    """
    Test handling when multiple matching times are found when end_time is not specified.
    """
    start_time = datetime(2022, 1, 1)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        global_dataset_with_multiple_times_per_day.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        with pytest.raises(
            ValueError,
            match="There must be exactly one time matching the start_time. Found 2 matching times.",
        ):
            Dataset(
                filename=filepath,
                var_names={"var": "var"},
                start_time=start_time,
                use_dask=use_dask,
            )
    finally:
        os.remove(filepath)


def test_warnings_times(global_dataset, use_dask):
    """
    Test handling when no matching times are found.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        global_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        with pytest.warns(
            Warning, match="No records found at or before the start_time."
        ):
            start_time = datetime(2021, 1, 1)
            end_time = datetime(2021, 2, 1)

            Dataset(
                filename=filepath,
                var_names={"var": "var"},
                start_time=start_time,
                end_time=end_time,
                use_dask=use_dask,
            )

        with pytest.warns(Warning, match="No records found at or after the end_time."):
            start_time = datetime(2024, 1, 1)
            end_time = datetime(2024, 2, 1)

            Dataset(
                filename=filepath,
                var_names={"var": "var"},
                start_time=start_time,
                end_time=end_time,
                use_dask=use_dask,
            )
    finally:
        os.remove(filepath)


def test_reverse_latitude_choose_subdomain_negative_depth(global_dataset, use_dask):
    """
    Test reversing latitude when it is not ascending, the choose_subdomain method, and the convert_to_negative_depth method of the Dataset class.
    """
    start_time = datetime(2022, 1, 1)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        global_dataset["latitude"] = global_dataset["latitude"][::-1]
        global_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(
            filename=filepath,
            var_names={"var": "var"},
            dim_names={
                "latitude": "latitude",
                "longitude": "longitude",
                "time": "time",
                "depth": "depth",
            },
            start_time=start_time,
            use_dask=use_dask,
        )

        assert np.all(np.diff(dataset.ds["latitude"]) > 0)

        # test choosing subdomain for domain that straddles the dateline
        dataset.choose_subdomain(
            latitude_range=(-10, 10), longitude_range=(-10, 10), margin=1, straddle=True
        )

        assert -11 <= dataset.ds["latitude"].min() <= 11
        assert -11 <= dataset.ds["latitude"].max() <= 11
        assert -11 <= dataset.ds["longitude"].min() <= 11
        assert -11 <= dataset.ds["longitude"].max() <= 11

        # test choosing subdomain for domain that does not straddle the dateline
        dataset = Dataset(
            filename=filepath,
            var_names={"var": "var"},
            dim_names={
                "latitude": "latitude",
                "longitude": "longitude",
                "time": "time",
                "depth": "depth",
            },
            start_time=start_time,
            use_dask=use_dask,
        )
        dataset.choose_subdomain(
            latitude_range=(-10, 10), longitude_range=(10, 20), margin=1, straddle=False
        )

        assert -11 <= dataset.ds["latitude"].min() <= 11
        assert -11 <= dataset.ds["latitude"].max() <= 11
        assert 9 <= dataset.ds["longitude"].min() <= 21
        assert 9 <= dataset.ds["longitude"].max() <= 21

        dataset.convert_to_negative_depth()

        assert (dataset.ds["depth"] <= 0).all()

    finally:
        os.remove(filepath)


def test_check_if_global_with_global_dataset(global_dataset, use_dask):

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        global_dataset.to_netcdf(filepath)
    try:
        dataset = Dataset(
            filename=filepath, var_names={"var": "var"}, use_dask=use_dask
        )
        is_global = dataset.check_if_global(dataset.ds)
        assert is_global
    finally:
        os.remove(filepath)


def test_check_if_global_with_non_global_dataset(non_global_dataset, use_dask):

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        non_global_dataset.to_netcdf(filepath)
    try:
        dataset = Dataset(
            filename=filepath, var_names={"var": "var"}, use_dask=use_dask
        )
        is_global = dataset.check_if_global(dataset.ds)

        assert not is_global
    finally:
        os.remove(filepath)


def test_check_dataset(global_dataset, use_dask):

    ds = global_dataset.copy()
    ds = ds.drop_vars("var")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        ds.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        start_time = datetime(2022, 2, 1)
        end_time = datetime(2022, 3, 1)
        with pytest.raises(
            ValueError, match="Dataset does not contain all required variables."
        ):

            Dataset(
                filename=filepath,
                var_names={"var": "var"},
                start_time=start_time,
                end_time=end_time,
                use_dask=use_dask,
            )
    finally:
        os.remove(filepath)

    ds = global_dataset.copy()
    ds = ds.rename({"latitude": "lat", "longitude": "long"})

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        ds.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        start_time = datetime(2022, 2, 1)
        end_time = datetime(2022, 3, 1)
        with pytest.raises(
            ValueError, match="Dataset does not contain all required dimensions."
        ):

            Dataset(
                filename=filepath,
                var_names={"var": "var"},
                start_time=start_time,
                end_time=end_time,
                use_dask=use_dask,
            )
    finally:
        os.remove(filepath)


def test_era5_correction_choose_subdomain(use_dask):

    data = ERA5Correction(use_dask=use_dask)
    lats = data.ds.latitude[10:20]
    lons = data.ds.longitude[10:20]
    coords = {"latitude": lats, "longitude": lons}
    data.choose_subdomain(coords, straddle=False)
    assert (data.ds["latitude"] == lats).all()
    assert (data.ds["longitude"] == lons).all()


def test_data_concatenation(use_dask):

    fname = download_test_data("GLORYS_NA_2012.nc")
    data = GLORYSDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )

    # Concatenating the datasets at fname0 and fname1 should result in the dataset at fname
    fname0 = download_test_data("GLORYS_NA_20120101.nc")
    fname1 = download_test_data("GLORYS_NA_20121231.nc")

    # Test concatenation based on wildcards
    directory_path = Path(fname0).parent
    data_concatenated = GLORYSDataset(
        filename=str(directory_path) + "/GLORYS_NA_2012????.nc",
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    assert data.ds.equals(data_concatenated.ds)

    # Test concatenation based on lists
    data_concatenated = GLORYSDataset(
        filename=[fname0, fname1],
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    assert data.ds.equals(data_concatenated.ds)
