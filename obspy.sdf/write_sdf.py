#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prototype implementation for a new file format using Python and ObsPy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import argparse
import copy
import h5py
import io
import numpy as np
import obspy
import os
import warnings
import sys


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
            if "sdf_format_version" not in self.__file.attrs:
                msg = ("No file format version given for file '%s'. The function "
                       "will continue but the result is undefined." %
                       self.__file.filename)
                warnings.warn(msg, SDFWarnings)
            elif self.__file.attrs["sdf_format_version"] != self.__file.attrs:
                msg = ("The file '%s' has version number '%s'. The reader "
                       "expects version '%s'. The function will continue but "
                       "the result is undefined." % (
                    self.__file.filename,
                    self.__file.attrs["sdf_format_version"],
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


    def __del__(self):
        """
        Attempts to close the HDF5 file.
        """
        try:
            self.__file.close()
        except:
            pass

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
            # Actually add the data.
            station_group.create_dataset(
                data_name, data=trace.data, compression=self.__compression[0],
                compression_opts=self.__compression[1], fletcher32=True)

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
                        existing_station_xml.networks[0].station[0]
                    # XXX: Need better checks for duplicates...
                    for channel in station.channels:
                        if channel in existing_channels:
                            continue
                        existing_channels.append(channel)
                    new_station_xml = existing_station_xml
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
                station_group.create_dataset("StationXML",
                                             data=np.void(temp.read()),
                                             fletcher32=True)
                temp.close()


    def add_quakeml(self, quakeml):
        pass


def _generate_unique_name(station_name, starttime, endtime, existing_names,
                          tag=None):
    """
    Helper function for creating the names of the traces. This ensures they are
    unique.
    """
    base_name = "{station_name}_{starttime}_{endtime}{tag}".format(
        station_name=station_name,
        starttime=starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        endtime=endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        tag=tag if tag else "")
    extra_tag = 0
    while True:
        if extra_tag:
            base_name += "_%i" % extra_tag
        if base_name not in existing_names:
            break
        extra_tag += 1
    return base_name


def write_sdf(stream, file_object, append=False, compression="szip-nn-10",
              tag=None):
    """
    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The stream to be written to the filename.
    :type file_object: filename or open h5py object.
    :param file_object: The filename or object to be written to.
    :type append: bool, optional
    :param append: If False, a new file will be created. If True, the data will
        be added to an possibly already existing file. Defaults to False. Only
        useful when a filename is given.
    :type compression: str, optional
    :param compression: The compression to use. Defaults to 'szip-nn-10' which
        yielded good results in the past.
    :type tag: basestring, optional
    :param tag: An additional tag to append to the name of each trace in the
        stream object.
    """
    if compression not in COMPRESSIONS:
        msg = "Unknown compressions '%s'. Available compressions: \n\t%s" % (
            compression, "\n\t".join(sorted(
                [str(i) for i in COMPRESSIONS.keys()])))
        raise Exception(msg)
    compression = COMPRESSIONS[compression]

    # Open file, either appending or truncating.
    if isinstance(file_object, basestring):
        if append is True:
            f = h5py.File(file_object, "a")
        else:
            f = h5py.File(file_object, "w")
    else:
        f = file_object

    # Write some attributes to the file.
    f.attrs["file_format"] = "SDF"
    f.attrs["file_format_version"] = FORMAT_VERSION

    # Get waveforms, create if non existent.
    if not "Waveforms" in f:
        f.create_group("Waveforms")
    waveforms = f["Waveforms"]

    # Loop over traces and add them.
    for trace in stream:
        station_name = ".".join((trace.stats.network, trace.stats.station))
        if not station_name in waveforms:
            waveforms.create_group(station_name)
        station = waveforms[station_name]
        # Add the data.
        data_name = _generate_unique_name(station_name, trace.stats.starttime,
                                          trace.stats.endtime, station.keys(),
                                          tag=tag)
        data = station.create_dataset(data_name, shape=trace.data.shape,
                                      dtype=trace.data.dtype,
                                      compression=compression[0],
                                      compression_opts=compression[1])
        # Actually write the data.
        data[...] = trace.data

    # Close if created in the function.
    if isinstance(file_object, basestring):
        f.close()


def add_quakeml(file_object, quakeml_filename):
    """
    Add a QuakeML file to an existing open HDF5 object.
    """
    f = file_object
    if "QuakeML" in f:
        msg = "HDF5 file already contains a QuakeML file"
        raise Exception(msg)

    with open(quakeml_filename, "rb") as fh:
        quake_ml_array = np.void(fh.read())

    quake_str = f.create_dataset("QuakeML", shape=quake_ml_array.shape,
                                 dtype=quake_ml_array.dtype)
    quake_str[...] = quake_ml_array


def add_stationxml(file_object, stationxml_filename):
    """
    Traverses the HDF5 file and adds the station where appropriate. Currently
    parses the station filename to get the station. Should of course be handled
    differently.
    """
    f = file_object

    # Get waveforms, create if non existent.
    if not "Waveforms" in f:
        f.create_group("Waveforms")
    waveforms = f["Waveforms"]

    station_name = ".".join(os.path.basename(
        stationxml_filename).split(".")[:2])

    if not station_name in waveforms:
        waveforms.create_group(station_name)
    station = waveforms[station_name]

    if "StationXML" in station:
        msg = "HDF5 file already contains a StationXML file for station %s." \
            % station_name
        raise warnings.warn(msg)
        return

    with open(stationxml_filename, "rb") as fh:
        station_array = np.void(fh.read())

    station_str = station.create_dataset(
        "StationXML", shape=station_array.shape, dtype=station_array.dtype)
    station_str[...] = station_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a big SDF file from a QuakeML file, a folder of "
        "waveform files, and a folder of StationXML files. Only useful for "
        "event based data right now.")
    parser.add_argument("--quakeml", required=True, dest="quakeml",
                        help="The QuakeML file to be used.")
    parser.add_argument("--stationxml_path", required=True, dest="station_xml",
                        help="Path a folder containing only StationXML files.")
    parser.add_argument("--waveform_path", required=True, dest="waveforms",
                        help="Path a folder containing only waveform files "
                        "belonging to the event.")
    parser.add_argument("-o", "--output", required=True, dest="output",
                        help="Output filename.")
    args = parser.parse_args()

    # Limited sanity checks.
    if not os.path.exists(args.station_xml) or \
            not os.path.isdir(args.station_xml):
        msg = "StationXML folder does not exist."
        raise Exception(msg)
    if not os.path.exists(args.waveforms) or not os.path.isdir(args.waveforms):
        msg = "Waveform folder does not exist."
        raise Exception(msg)
    if not os.path.exists(args.quakeml) or not os.path.isfile(args.quakeml):
        msg = "QuakeML file does not exist."
        raise Exception(msg)
    if os.path.exists(args.output):
        msg = "Output path already exists."
        raise Exception(msg)

    file_object = h5py.File(args.output, "w")
    add_quakeml(file_object, args.quakeml)

    print "Adding waveforms..."
    for filename in os.listdir(args.waveforms):
        # Make sure it is written for every file.
        sys.stdout.write(".")
        sys.stdout.flush()
        try:
            st = obspy.read(os.path.join(args.waveforms, filename))
        except:
            msg = "'%s' could not be read." % filename
            warnings.warn(msg)
            continue
        write_sdf(st, file_object, append=True)
    print ""

    print "Adding stations..."
    for filename in os.listdir(args.station_xml):
        # Make sure it is written for every file.
        sys.stdout.write(".")
        sys.stdout.flush()
        add_stationxml(file_object, os.path.join(args.station_xml, filename))
    print ""

    file_object.close()
