# SDF


### HDF5 prototype realized with ObsPy

The folder `obspy.sdf` contains a very rough prototype realized using [ObsPy](http://obspy.org)
and [h5py](http://www.h5py.org/). It is not parallel but that should not be a big issue once
a nice interface has been designed, see [here](http://www.h5py.org/docs/topics/mpi.html).

It has a very simple command line interface:

```bash
$ python write_sdf.py --help

usage: write_sdf.py [-h] --quakeml QUAKEML --stationxml_path STATION_XML
                    --waveform_path WAVEFORMS -o OUTPUT

Create a big SDF file from a QuakeML file, a folder of waveform files, and a
folder of StationXML files. Only useful for event based data right now.

optional arguments:
  -h, --help            show this help message and exit
  --quakeml QUAKEML     The QuakeML file to be used.
  --stationxml_path STATION_XML
                        Path a folder containing only StationXML files.
  --waveform_path WAVEFORMS
                        Path a folder containing only waveform files belonging
                        to the event.
  -o OUTPUT, --output OUTPUT
                        Output filename.
```
