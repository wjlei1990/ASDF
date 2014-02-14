# SDF


### HDF5 prototype realized with ObsPy

The folder `obspy.sdf` contains a very rough prototype realized using [ObsPy](http://obspy.org)
and [h5py](http://www.h5py.org/).

It currently contains a data set class that has a nice interface to manipulate data. All operations
are transparently mapped to an HDF5 file on the disc. This is not done yet and should rather be 
considered a proof of concept.

#### Creation of a new file

This section quickly introduces the creation of a new SDF file.

```python
from write_sdf import SDFDataSet

# If the file does not exist, it will be created,
# otherwise the old one will used.
data_set = SDFDataSet("test_file.h5")
```

One can add waveform and station data in any order. The class will take care that everything is
mapped to the correct section in the HDF5 file. The use of ObsPy means that we get a converter
for effectively all seismic waveform formats in use to the new format for free.

```
import glob

for filename in glob.glob("*.mseed"):
    data_set.add_waveform_file(filename, tag="raw_recording")
    
for filename in glob.glob("*.xml"):
    data_set.add_stationxml(filename)
```

It is also possible to do this with an already existing file. HDF5 is flexible enough.


#### Accessing data

This interactive session demonstrates how to use the class to access the data.

```python
>>> data_set = SDFDataSet("test_file.h5")
```

One can print some information.
```python
>>> print data_set
SDF file: 'test_file.h5' (2.7 GB)
        Contains data from 1392 stations.
```

And one can access waveforms and station. Tab completion works just fine. What comes back are ObsPy
objects which should enable a convenient way of working with the data and outputting it to any other
format.

The waveforms can be accessed via the tags. The return type is an ObsPy Stream object which will be
created on the fly when accessing it. This is essence enables one to work with huge datasets on 
a laptop as only the part of the data required at the moment is in memory.

```python
>>> st = data_set.waveforms.AE_113A.raw_recording
>>> print st
AE.113A..BHE | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
AE.113A..BHN | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
AE.113A..BHZ | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
>>> st.plot()
```

The same is true with the station information which return an ObsPy inventory object.
```python
>>> inv = data_set.waveforms.AE_113A.StationXML
>>> print inv
Inventory created at 2014-02-08T22:06:43.000000Z
        Created by: IRIS WEB SERVICE: fdsnws-station | version: 1.0.10
                    http://service.iris.edu/fdsnws/station/1/query?channel=BH%2A&statio...
        Sending institution: IRIS-DMC (IRIS-DMC)
        Contains:
                Networks (1):
                        AE
                Stations (1):
                        AE.113A (Mohawk Valley, Roll, AZ, USA)
                Channels (3):
                        AE.113A..BHE, AE.113A..BHN, AE.113A..BHZ
```

So now one has all the information needed to process the data. The following snippet will convert all
data for the given station and tag to meters per second.

```python
>>> st.attach_response(inv)
>>> st.remove_response(units="VEL")
```


#### Large Scale Processing

This is not yet fully implemented but will be done soon. So the idea is to define a function per station and tag. This function will then be applied to all data and the result will be stored in a new file. If an MPI environment is detected it will be distributed across all nodes and otherwise `os.fork` will be used for shared memory multiprocessing. This should all happen behind the scenes and the user does not have to bother with it.

```python
def process(stream, station):
    stream.attach_resonse(station)
    stream.remove_response(units="VEL")
    
data_set(process, output_filename="new.h5")
```
