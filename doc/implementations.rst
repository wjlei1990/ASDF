Implementations
===============

Fortran API
-----------

The Fortran API is intended to be used to generate ASDF output files in numerical wave propagation codes like SPECFEM. This will result in only one file for an arbitrary number of receiving stations.

Once the results have been produced, the same file can be analyzed and processed with Python.

Python API
---------------

This implementation is currently a rough protoytype realized using `ObsPy
<http://obspy.org>`_ and `h5py <http://www.h5py.org>`_. Usability while
retaining acceptable performance is the key design goal for this
implementation.

It is based around a data set class that has a nice interface to manipulate
data. All operations are transparently mapped to an HDF5 file on disc. This
means that only as little data as necessary is held in memory at any time. As
soon as some piece of data has finished processing it will be written to the
file.  This enables one to process arbitrarily large files on personal
computers and laptops.

Converting Data to ASDF
^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    The use of ObsPy means that we get a converter for effectively all seismic
    waveform data formats for free.


This section demonstrates the creation of a new ASDF file from existing data.
The first step is to initialize a new data set object with a filename. If the
file does not yet exist, it will be created, otherwise it will be read and
appended if necessary.

.. code-block:: python

    from obspy_asdf import ASDFDataSet

    # If the file does not exist, it will be created,
    # otherwise the old one will used.
    data_set = ASDFDataSet("test_file.h5")

One can add waveform and station data in any order. The class will take care
that everything is mapped to the correct groups in the HDF5 file.

.. code-block:: python

    import glob

    # The add_waveform_file() method is able to work with filenamen, open
    # files, memory files, web adresses and ObsPy objects.
    for filename in glob.glob("*.mseed"):
        data_set.add_waveform_file(filename, tag="raw_recording")

    for filename in glob.glob("*.xml"):
        data_set.add_stationxml(filename)

It is also possible to do this with an already existing file - HDF5 is flexible
enough to allow this. The methods will warn when trying to add already existing
data or metadata.


Accessing data
^^^^^^^^^^^^^^

This interactive session demonstrates how to use the class to access the data.

.. code-block:: python

    >>> data_set = ASDFDataSet("test_file.h5")

One can print some information.

.. code-block:: python

    >>> print data_set
    ASDF file: 'test_file.h5' (2.7 GB)
        Contains data from 1392 stations.

And one can access waveforms and station. Tab completion works just fine. What
comes back are ObsPy objects which should enable a convenient way of working
with the data and outputting it to any other format.

The waveforms can be accessed via the tags. The return type is an ObsPy Stream
object which will be created on the fly when accessing it. This is essence
enables one to work with huge datasets on a laptop as only the part of the data
required at the moment is in memory.

.. code-block:: python

    >>> st = data_set.waveforms.AE_113A.raw_recording
    >>> print st
    AE.113A..BHE | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
    AE.113A..BHN | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
    AE.113A..BHZ | 2013-05-24T05:40:00.000000Z - 2013-05-24T06:50:00.000000Z | 40.0 Hz, 168001 samples
    >>> st.plot()

The same is true with the station information which return an ObsPy inventory
object.

.. code-block:: python

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

So now one has all the information needed to process the data. The following
snippet will convert all data for the given station and tag to meters per
second.

.. code-block:: python

    >>> st.attach_response(inv)
    >>> st.remove_response(units="VEL")


Large Scale Processing
^^^^^^^^^^^^^^^^^^^^^^

This is currently just a collection of ideas and some details must be figured
out when implementing it.

This is not yet fully implemented but will be done soon. So the idea is to
define a function per station and tag. This function will then be applied to
all data and the result will be stored in a new file. If an MPI environment is
detected it will be distributed across all nodes and otherwise `os.fork` will
be used for shared memory multiprocessing. This should all happen behind the
scenes and the user does not have to bother with it.

.. note::

    This snippet of code should work regardless of the environment. If an MPI
    environment is detected it will be used, otherwise forked processes will be
    employed.

.. code-block:: python

    def process(stream, station):
        stream.attach_resonse(station)
        stream.remove_response(units="VEL")

    data_set.map(process, output_filename="new.h5")

The output is a new file, with all stations being processed by the defined function.

A similar interface will be offered when combining two files. In this case one
function should be called for each station containing the data from both files.


.. code-block:: python

    # First data set
    data_set_1 = ...

    # Second data set
    data_set_2 = ...

    def matched_processing(stream, station, other_stream, other_station):
        diffs = []
        for trace, other_trace in zip(stream, other_stream):
            diff = trace.data - other_trace.data
        # The returned values will be somehow saved to the file.
        return {"ret_type": "Difference", "data": diffs}

    data_set_1.map_matched(data_set_2, matched_processing,
                           output_filename="new.h5")


Other interfaces will likely become necessary with time and will be implemented
once the need becomes clear.
