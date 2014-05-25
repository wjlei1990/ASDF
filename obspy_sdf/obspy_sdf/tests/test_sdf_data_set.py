import glob
import inspect
import shutil
import obspy
import os
import pytest

from obspy_sdf import SDFDataSet

data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

class Namespace(object):
    """
    Simple helper class offering a namespace.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def example_data_set(tmpdir):
    """
    Fixture creating a small example file.
    """
    sdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = SDFDataSet(sdf_filename)

    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveform_file(filename, tag="raw_recording")

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            continue
        data_set.add_stationxml(filename)

    # Flush and finish writing.
    del data_set

    # Return filename and path to tempdir, no need to always create a
    # new one.
    return Namespace(filename=sdf_filename, tmpdir=tmpdir.strpath)


def test_data_set_creation(tmpdir):
    """
    Test data set creation with a small test dataset.

    It tests that the the stuff that goes in is correctly saved and
    can be retrieved again.
    """
    sdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = SDFDataSet(sdf_filename)

    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveform_file(filename, tag="raw_recording")

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            continue
        data_set.add_stationxml(filename)

    # Flush and finish writing.
    del data_set

    # Open once again
    data_set = SDFDataSet(sdf_filename)
    data_set == data_set

    # ObsPy is tested enough to make this comparison meaningful.
    for station in (("AE", "113A"), ("TA", "POKR")):
        # Test the waveforms
        stream_sdf = \
            getattr(data_set.waveforms, "%s_%s" % station).raw_recording
        stream_file = obspy.read(os.path.join(
            data_path, "%s.%s.*.mseed" % station))
        # Delete the file format specific stats attributes.
        for trace in stream_file:
            del trace.stats.mseed
            del trace.stats._format

        # Now the Inventory data.
        inv_sdf = \
            getattr(data_set.waveforms, "%s_%s" % station).StationXML
        inv_file = obspy.read_inventory(
            os.path.join(data_path, "%s.%s..BH*.xml" % station))
        assert inv_file == inv_sdf


def test_equality_checks(example_data_set):
    """
    Tests the equality operations.
    """
    filename_1 = example_data_set.filename
    filename_2 = os.path.join(example_data_set.tmpdir, "new.h5")
    shutil.copyfile(filename_1, filename_2)

    data_set_1 = SDFDataSet(filename_1)
    data_set_2 = SDFDataSet(filename_2)

    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # A tiny change at an arbitrary place should trigger an inequality.
    for tag, data_array in data_set_2._waveform_group["AE.113A"].items():
        if tag != "StationXML":
            break
    data_array[1] += 2.0
    assert not (data_set_1 == data_set_2)
    assert data_set_1 != data_set_2

    # Reverting should also work. Floating point math inaccuracies should
    # not matter at is only tests almost equality. This is not a proper test
    # for this behaviour though.
    data_array[1] -= 2.0
    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # Also check the StationXML.
    data_array = data_set_2._waveform_group["AE.113A"]["StationXML"]
    data_array[1] += 2.0
    assert not (data_set_1 == data_set_2)
    assert data_set_1 != data_set_2
    data_array[1] -= 2.0
    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # Test change of keys.
    del data_set_1._waveform_group["AE.113A"]
    assert data_set_1 != data_set_2
