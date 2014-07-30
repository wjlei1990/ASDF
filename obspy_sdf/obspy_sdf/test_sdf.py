import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools

from obspy_sdf import SDFDataSet


filename = "ccs.h5"
data_set = SDFDataSet(filename)


stations = [("TA", "BA"), ("BW", "FURT"), ("HEL", "LO")]


for station_1, station_2 in itertools.combinations(stations, 2):
    print station_1, station_2

# for station_pair in station_pair_generator():
#     cc = get_cc(station_pair)
#     data_set.add_auxiliary_data(cc, data_type="cross_correlation",
#                               tag=station_pair.get_tag(),
#                               options={"window_length": ...,
#                                        "correlation_type", ...})



# print "Adding MiniSEED files"
# for file in glob.iglob(os.path.join(path, "MiniSEED", "*.mseed")):
#     print file
#     data_set.add_waveforms(file, tag="raw_recording")
#
# print "Adding StationXML files"
# for file in glob.iglob(os.path.join(path, "StationXML", "*.xml")):
#     print file
#     data_set.add_stationxml(file)
