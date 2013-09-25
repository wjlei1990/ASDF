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
import h5py
import obspy
import os
import warnings


FORMAT_VERSION = "0.0.1a"
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


def write_sdf(stream, filename, append=False, compression="szip-nn-10",
              tag=None):
    """
    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The stream to be written to the filename.
    :type filename: basestring
    :param filename: The filename to be written to.
    :type append: bool, optional
    :param append: If False, a new file will be created. If True, the data will
        be added to an possibly already existing file. Defaults to False.
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
    # Open file, either appending or truncating.
    if append is True:
        f = h5py.File(filename, "a")
    else:
        f = h5py.File(filename, "w")

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
    f.close()


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

    for filename in os.listdir(args.waveforms):
        print ".",
        try:
            st = obspy.read(os.path.join(args.waveforms, filename))
        except:
            msg = "'%s' could not be read." % filename
            warnings.warn(msg)
            continue

        # This is rather inefficient as it will open and close the file a lot.
        # But just for testing it is fine!
        write_sdf(st, args.output, append=True)
    print ""
