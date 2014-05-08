import glob
import inspect
import obspy
import os

from obspy_sdf import SDFDataSet

data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


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
        # Then they should be equal.
        assert stream_sdf == stream_file

        # Now the Inventory data.
        inv_sdf = \
            getattr(data_set.waveforms, "%s_%s" % station).StationXML
        inv_file = obspy.read_inventory(
            os.path.join(data_path, "%s.%s..BH*.xml" % station))
        assert inv_file == inv_sdf
