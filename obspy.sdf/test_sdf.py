import glob
import os

from write_sdf import SDFDataSet

path = "/Users/lion/workspace/data/2014_01--SDF_Earthquake_data_set"
filename = "test_file.h5"

if os.path.exists(filename):
    os.remove(filename)


data_set = SDFDataSet(filename)

# for file in glob.iglob(os.path.join(path, "MiniSEED", "*.mseed")):
#     data_set.add_waveform_file(file, tag="raw_recording")

for file in glob.iglob(os.path.join(path, "StationXML", "*.xml")):
    print file
    data_set.add_stationxml(file)
