import glob
import os

from write_sdf import SDFDataSet

path = "/Users/lion/workspace/data/2014_01--SDF_Earthquake_data_set"
filename = "test_file.h5"


data_set = SDFDataSet(filename)
print ""
print data_set
print ""


def process_function(stream, inventory):
    stream.detrend("linear")
    stream.decimate(factor=5
    stream.attach_response(inventory)
    stream.remove_response(units="velocity")
    stream.filter("lowpass", 2.0)


# Apply this to all stations. This will in the future detect if is it run in
# an MPI environment and otherwise use os.fork to achieve parallelism.
data_set.process(process_function, output_filename="new.h5")


# print "Adding MiniSEED files"
# for file in glob.iglob(os.path.join(path, "MiniSEED", "*.mseed")):
#     print file
#     data_set.add_waveform_file(file, tag="raw_recording")
#
# print "Adding StationXML files"
# for file in glob.iglob(os.path.join(path, "StationXML", "*.xml")):
#     print file
#     data_set.add_stationxml(file)
