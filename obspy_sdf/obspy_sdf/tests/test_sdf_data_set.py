import glob
import h5py
import inspect
import io
import mock
import shutil
import obspy
import os
import pytest

from obspy_sdf import SDFDataSet
from obspy_sdf.header import FORMAT_VERSION, FORMAT_NAME


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
        data_set.add_waveforms(filename, tag="raw_recording")

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            data_set.add_quakeml(filename)
        else:
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
        data_set.add_waveforms(filename, tag="raw_recording")

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            data_set.add_quakeml(filename)
        else:
            data_set.add_stationxml(filename)

    # Flush and finish writing.
    del data_set

    # Open once again
    data_set = SDFDataSet(sdf_filename)

    # ObsPy is tested enough to make this comparison meaningful.
    for station in (("AE", "113A"), ("TA", "POKR")):
        # Test the waveforms
        stream_sdf = \
            getattr(data_set.waveforms, "%s_%s" % station).raw_recording
        stream_file = obspy.read(os.path.join(
            data_path, "%s.%s.*.mseed" % station))
        # Delete the file format specific stats attributes. These are
        # meaningless inside SDF data sets.
        for trace in stream_file:
            del trace.stats.mseed
            del trace.stats._format
        for trace in stream_sdf:
            del trace.stats.sdf
            del trace.stats._format
        assert stream_sdf == stream_file

        # Test the inventory data.
        inv_sdf = \
            getattr(data_set.waveforms, "%s_%s" % station).StationXML
        inv_file = obspy.read_inventory(
            os.path.join(data_path, "%s.%s..BH*.xml" % station))
        assert inv_file == inv_sdf
    # Test the event.
    cat_file = obspy.readEvents(os.path.join(data_path, "quake.xml"))
    cat_sdf = data_set.events
    # from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
    assert cat_file == cat_sdf


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


def test_adding_same_event_twice_raises(tmpdir):
    """
    Adding the same event twice raises.
    """
    sdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = SDFDataSet(sdf_filename)

    # Add once, all good.
    data_set.add_quakeml(os.path.join(data_path, "quake.xml"))
    assert len(data_set.events) == 1

    # Adding again should raise an error.
    with pytest.raises(ValueError):
        data_set.add_quakeml(os.path.join(data_path, "quake.xml"))


def test_adding_event_in_various_manners(tmpdir):
    """
    Events can be added either as filenames, open files, BytesIOs, or ObsPy
    objects. In any case, the result should be the same.
    """
    sdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")
    event_filename = os.path.join(data_path, "quake.xml")

    ref_cat = obspy.readEvents(event_filename)

    # Add as filename
    data_set = SDFDataSet(sdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(event_filename)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(sdf_filename)

    # Add as open file.
    data_set = SDFDataSet(sdf_filename)
    assert len(data_set.events) == 0
    with open(event_filename, "rb") as fh:
        data_set.add_quakeml(fh)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(sdf_filename)

    # Add as BytesIO.
    data_set = SDFDataSet(sdf_filename)
    assert len(data_set.events) == 0
    with open(event_filename, "rb") as fh:
        temp = io.BytesIO(fh.read())
    temp.seek(0, 0)
    data_set.add_quakeml(temp)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(sdf_filename)

    # Add as ObsPy Catalog.
    data_set = SDFDataSet(sdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(ref_cat.copy())
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(sdf_filename)

    # Add as an ObsPy event.
    data_set = SDFDataSet(sdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(ref_cat.copy()[0])
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(sdf_filename)


def test_assert_format_and_version_number_are_written(tmpdir):
    """
    Check that the version number and file format name are correctly written.
    """
    sdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    # Create empty data set.
    data_set = SDFDataSet(sdf_filename)
    # Flush and write.
    del data_set

    # Open again and assert name and version number.
    with h5py.File(sdf_filename, "r") as hdf5_file:
        assert hdf5_file.attrs["file_format_version"] == FORMAT_VERSION
        assert hdf5_file.attrs["file_format"] == FORMAT_NAME


def test_dot_accessors(example_data_set):
    """
    Tests the dot accessors for waveforms and stations.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    data_set = SDFDataSet(example_data_set.filename)

    # Get the contents, this also asserts that tab completions works.
    assert sorted(dir(data_set.waveforms)) == ["AE_113A", "TA_POKR"]
    assert sorted(dir(data_set.waveforms.AE_113A))  == \
        ["StationXML", "raw_recording"]
    assert sorted(dir(data_set.waveforms.TA_POKR))  == \
        ["StationXML", "raw_recording"]

    # Actually check the contents.
    waveform = data_set.waveforms.AE_113A.raw_recording.sort()
    waveform_file = obspy.read(os.path.join(data_path, "AE.*.mseed")).sort()
    for trace in waveform_file:
        del trace.stats.mseed
        del trace.stats._format
    for trace in waveform:
        del trace.stats.sdf
        del trace.stats._format
    assert waveform == waveform_file

    waveform = data_set.waveforms.TA_POKR.raw_recording.sort()
    waveform_file = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    for trace in waveform_file:
        del trace.stats.mseed
        del trace.stats._format
    for trace in waveform:
        del trace.stats.sdf
        del trace.stats._format
    assert waveform == waveform_file

    assert data_set.waveforms.AE_113A.StationXML == \
       obspy.read_inventory(os.path.join(data_path, "AE.113A..BH*.xml"))
    assert data_set.waveforms.TA_POKR.StationXML == \
           obspy.read_inventory(os.path.join(data_path, "TA.POKR..BH*.xml"))


def test_stationxml_is_invalid_tag_name(tmpdir):
    """
    StationXML is an invalid waveform tag.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = SDFDataSet(filename)
    st = obspy.read()

    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="StationXML")
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="stationxml")

    # Adding with a proper tag works just fine.
    data_set.add_waveforms(st, tag="random_waveform")


def test_saving_event_id(tmpdir):
    """
    Tests that the event_id can be saved and retrieved automatically.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    filename = os.path.join(tmpdir.strpath, "example.h5")
    event = obspy.readEvents(os.path.join(data_path, "quake.xml"))[0]

    # Add the event object, and associate the waveform with it.
    data_set = SDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording", event_id=event)
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.sdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)

    # Add as a string.
    data_set = SDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording",
                           event_id=str(event.resource_id.id))
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.sdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)

    # Add as a resource identifier object.
    data_set = SDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording",
                           event_id=event.resource_id)
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.sdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)
