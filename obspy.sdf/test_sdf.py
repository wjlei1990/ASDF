import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import glob

from sdf_data_set import SDFDataSet

path = "/Users/lion/workspace/data/2014_01--SDF_Earthquake_data_set"
filename = "test_file.h5"


data_set = SDFDataSet(filename)
if not data_set.mpi or data_set.mpi.rank == 0:
    print(data_set)
    print("")


import time
import random

def process_function(stream, inventory):
    time.sleep(random.random() * 10)
    # stream.detrend("linear")
    # stream.decimate(factor=5)
    # stream.attach_response(inventory)
    # stream.remove_response(output="VEL")
    # stream.filter("lowpass", freq=2.0)

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
