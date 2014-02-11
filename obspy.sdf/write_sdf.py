#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prototype implementation for a new file format using Python and ObsPy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import copy
import h5py
import io
import numpy as np
import obspy
import os
import warnings
import weakref


FORMAT_NAME = "SDF"
FORMAT_VERSION = "0.0.1b"


# List all compression options.
COMPRESSIONS = {
    None: (None, None),
    "lzf": ("lzf", None),
    "gzip-0": ("gzip", 0),
    "gzip-1": ("gzip", 1),
    "gzip-2": ("gzip", 2),
    "gzip-3": ("gzip", 3),
    "gzip-4": ("gzip", 4),
    "gzip-5": ("gzip", 5),
    "gzip-6": ("gzip", 6),
    "gzip-7": ("gzip", 7),
    "gzip8-": ("gzip", 8),
    "gzip-9": ("gzip", 9),
    "szip-ec-8": ("szip", ("ec", 8)),
    "szip-ec-10": ("szip", ("ec", 10)),
    "szip-nn-8": ("szip", ("nn", 8)),
    "szip-nn-10": ("szip", ("nn", 10))
}


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesize.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


class SDFException(Exception):
    """
    Generic exception for the Python SDF implementation.
    """
    pass


class SDFWarnings(UserWarning):
    """
    Generic SDF warning.
    """
    pass


class StationAccessor(object):
    """
    Helper class to facilitate access to the waveforms and stations.
    """
    def __init__(self, sdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(sdf_data_set)


    def __getattr__(self, item):
        __waveforms = self.__data_set()._waveform_group
        if item.replace("_", ".") not in __waveforms:
            raise AttributeError
        return WaveformAccessor(item.replace("_", "."), self.__data_set())

    def __dir__(self):
        __waveforms = self.__data_set()._waveform_group
        return [_i.replace(".", "_") for _i in __waveforms.iterkeys()]


class WaveformAccessor(object):
    """
    Helper class facilitating access to the actual waveforms and stations.
    """
    def __init__(self, station_name, sdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__station_name = station_name
        self.__data_set = weakref.ref(sdf_data_set)

    def __getattr__(self, item):
        __station = self.__data_set()._waveform_group[self.__station_name]
        keys = [_i for _i in __station.iterkeys()
            if _i.endswith("__" + item)]
        traces = [self.__data_set().get_waveform(_i) for _i in keys]
        return obspy.Stream(traces=traces)

    def __dir__(self):
        __station = self.__data_set()._waveform_group[self.__station_name]
        directory = []
        if "StationXML" in __station:
            directory.append("StationXML")
        directory.extend([_i.split("__")[-1]
                          for _i in __station.iterkeys()
                          if _i != "StationXML"])
        return directory


class SDFDataSet(object):
    """
    DataSet object holding
    """
    def __init__(self, file_object, compression=None):
        """
        :type file_object: filename or open h5py object.
        :param file_object: The filename or object to be written to.
        :type compression: str, optional
        :param compression: The compression to use. Defaults to 'szip-nn-10'
            which yielded good results in the past. Will only be applied to
            newly added data sets. Existing ones are not touched.
        """
        if compression not in COMPRESSIONS:
            msg = "Unknown compressions '%s'. Available compressions: \n\t%s" \
                % (compression, "\n\t".join(sorted(
                [str(i) for i in COMPRESSIONS.keys()])))
            raise Exception(msg)
        self.__compression = COMPRESSIONS[compression]

        # Open file or take an already open file object.
        if isinstance(file_object, h5py.File):
            self.__file = file_object
        else:
            self.__file = h5py.File(file_object, "a")

        if "file_format" in self.__file.attrs:
            if self.__file.attrs["file_format"] != FORMAT_NAME:
                # Cleanup and raise.
                self.__del__()
                msg = "Not a '%s' file." % FORMAT_NAME
                raise SDFException(msg)
            if "file_format_version" not in self.__file.attrs:
                msg = ("No file format version given for file '%s'. The function "
                       "will continue but the result is undefined." %
                       self.__file.filename)
                warnings.warn(msg, SDFWarnings)
            elif self.__file.attrs["file_format_version"] != FORMAT_VERSION:
                msg = ("The file '%s' has version number '%s'. The reader "
                       "expects version '%s'. The function will continue but "
                       "the result is undefined." % (
                    self.__file.filename,
                    self.__file.attrs["file_format_version"],
                    FORMAT_VERSION))
                warnings.warn(msg, SDFWarnings)
        else:
            self.__file.attrs["file_format"] = FORMAT_NAME
            self.__file.attrs["file_format_version"] = FORMAT_VERSION

        # Create the waveform and provenance groups.
        if not "Waveforms" in self.__file:
            self.__file.create_group("Waveforms")
        self.__waveforms = self.__file["Waveforms"]
        if not "Provenance" in self.__file:
            self.__file.create_group("Provenance")
        self.__provenance = self.__file["Provenance"]

        self.waveforms = StationAccessor(self)

    @property
    def _waveform_group(self):
        return self.__file["Waveforms"]

    def get_waveform(self, waveform_name):
        """
        Retrieves the waveform for a certain tag name as a Trace object.
        """
        network, station, location, channel = waveform_name.split(".")[:4]
        channel = channel[:channel.find("__")]
        data = self.__file["Waveforms"]["%s.%s" % (network, station)][
            waveform_name]
        tr = obspy.Trace(data=data.value)
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.location = location
        tr.stats.channel = channel
        return tr

    def __del__(self):
        """
        Attempts to close the HDF5 file.
        """
        try:
            self.__file.close()
        except:
            pass

    def __str__(self):
        filesize = sizeof_fmt(os.path.getsize(self.__file.filename))
        ret = "{format} file: '{filename}' ({size})".format(
            format=FORMAT_NAME,
            filename=self.__file.filename,
            size=filesize
        )
        ret += "\n\tContains data from {len} stations.".format(
            len=len(self.__file["Waveforms"])
        )
        return ret

    def __get_station_group(self, network_code, station_code):
        """
        Helper function.
        """
        station_name = "%s.%s" % (network_code, station_code)
        if not station_name in self.__waveforms:
            self.__waveforms.create_group(station_name)
        return self.__waveforms[station_name]


    def add_waveform_file(self, waveform, tag):
        """
        Adds one or more waveforms to the file.

        :param waveform: The waveform to add. Can either be an ObsPy Stream or
            Trace object or something ObsPy can read.
        :type tag: String
        :param tag: The tag that will be given to all waveform files. It is
            mandatory for all traces and facilitates identification of the data
            within one SDF volume.
        """
        # The next function expects some kind of iterable that yields traces.
        if isinstance(waveform, obspy.Trace):
            waveform = [waveform]
        elif isinstance(waveform, obspy.Stream):
            pass
        # Delegate to ObsPy's format/input detection.
        else:
            waveform = obspy.read(waveform)

        # Actually add the data.
        for trace in waveform:
            station_group = self.__get_station_group(trace.stats.network,
                                                     trace.stats.station)
            # Generate the name of the data within its station folder.
            data_name = "{net}.{sta}.{loc}.{cha}__{start}__{end}__{tag}".format(
                net=trace.stats.network,
                sta=trace.stats.station,
                loc=trace.stats.location,
                cha=trace.stats.channel,
                start=trace.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"),
                end=trace.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S"),
                tag=tag)
            if data_name in station_group:
                msg = "Data '%s' already exists in file. Will not be added!" % \
                    data_name
                warnings.warn(msg, SDFWarnings)
                continue
            # Actually add the data. Use maxshape to create an extendable data
            # set.
            station_group.create_dataset(
                data_name, data=trace.data, compression=self.__compression[0],
                compression_opts=self.__compression[1], fletcher32=True,
                maxshape=(None,))
            station_group[data_name].attrs["starttime"] = \
                trace.stats.starttime.timestamp
            station_group[data_name].attrs["sampling_rate"] = \
                trace.stats.sampling_rate

    def add_stationxml(self, stationxml):
        """
        """
        if isinstance(stationxml, obspy.station.Inventory):
            pass
        else:
            stationxml = obspy.read_inventory(stationxml, format="stationxml")

        for network in stationxml:
            network_code = network.code
            for station in network:
                station_code = station.code
                station_group = self.__get_station_group(network_code,
                                                         station_code)
                # Get any already existing StationXML file. This will always
                # only contain a single station!
                if "StationXML" in station_group:
                    existing_station_xml = obspy.read_inventory(
                        io.BytesIO(
                            station_group["StationXML"].value.tostring()),
                        format="stationxml")
                    # Only exactly one station acceptable.
                    if len(existing_station_xml.networks) != 1 or  \
                            len(existing_station_xml.networks[0].stations) \
                            != 1:
                        msg = ("The existing StationXML file for station "
                               "'%s.%s' does not contain exactly one station!"
                               % (network_code, station_code))
                        raise SDFException(msg)
                    existing_channels = \
                        existing_station_xml.networks[0].stations[0].channels
                    # XXX: Need better checks for duplicates...
                    found_new_channel = False
                    chan_essence = [(_i.code, _i.location_code, _i.start_date,
                                     _i.end_date) for _i in existing_channels]
                    for channel in station.channels:
                        essence = (channel.code, channel.location_code,
                                   channel.start_date, channel.end_date)
                        if essence in chan_essence:
                            continue
                        existing_channels.appends(channel)
                        found_new_channel = True

                    # Only write if something actually changed.
                    if found_new_channel is True:
                        temp = io.BytesIO()
                        existing_station_xml.write(temp, format="stationxml")
                        temp.seek(0, 0)
                        data = np.array(list(temp.read()), dtype="|S1")
                        data.dtype = np.int8
                        temp.close()
                        # maxshape takes care to create an extendable data set.
                        station_group["StationXML"].resize((len(data.data),))
                        station_group["StationXML"][:] = data
                else:
                    # Create a shallow copy of the network and add the channels
                    # to only have the channels of this station.
                    new_station_xml = copy.copy(stationxml)
                    new_station_xml.networks = [network]
                    new_station_xml.networks[0].stations = [station]
                    # Finally write it.
                    temp = io.BytesIO()
                    new_station_xml.write(temp, format="stationxml")
                    temp.seek(0, 0)
                    data = np.array(list(temp.read()), dtype="|S1")
                    data.dtype = np.int8
                    temp.close()
                    # maxshape takes care to create an extendable data set.
                    station_group.create_dataset(
                        "StationXML", data=data,
                        maxshape=(None,),
                        fletcher32=True)

    def add_quakeml(self, quakeml):
        pass
