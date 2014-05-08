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
import collections
import h5py
import io
import multiprocessing
import math
import numpy as np
import obspy
import os
from UserDict import DictMixin
import warnings
import weakref
import sys
import time


FORMAT_NAME = "SDF"
FORMAT_VERSION = "0.0.1b"

# MPI message tags used for communication.
MSG_TAGS = [
    "MASTER_FORCES_WRITE",
    "MASTER_SENDS_ITEM",
    "WORKER_REQUESTS_ITEM",
    "WORKER_DONE_WITH_ITEM",
    "WORKER_REQUESTS_WRITE",
    # Message send by the master to indicate everything has been processed.
    # Otherwise all workers will keep looping to be able to synchronize
    # metadata.
    "ALL_DONE",
]

# Convert to two-way dict.
MSG_TAGS = {msg: i  for i, msg in enumerate(MSG_TAGS)}
MSG_TAGS.update({value: key for key, value in MSG_TAGS.items()})

ReceivedMessage = collections.namedtuple("ReceivedMessage", ["data"])

POISON_PILL = "POISON_PILL"

MAX_MEMORY_PER_WORKER_IN_MB = 100

input_data_set_container = []
output_data_set_container = []


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
        if item != "StationXML":
            __station = self.__data_set()._waveform_group[self.__station_name]
            keys = [_i for _i in __station.iterkeys()
                if _i.endswith("__" + item)]
            traces = [self.__data_set().get_waveform(_i) for _i in keys]
            return obspy.Stream(traces=traces)
        else:
            return self.__data_set().get_station(self.__station_name)

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
    def __init__(self, file_object, compression=None, debug=False):
        """
        :type file_object: filename or open h5py object.
        :param file_object: The filename or object to be written to.
        :type compression: str, optional
        :param compression: The compression to use. Defaults to 'szip-nn-10'
            which yielded good results in the past. Will only be applied to
            newly added data sets. Existing ones are not touched.
        """
        self.debug = debug
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
        tr.stats.starttime = obspy.UTCDateTime(data.attrs["starttime"])
        tr.stats.sampling_rate = float(data.attrs["sampling_rate"])
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.location = location
        tr.stats.channel = channel
        return tr

    def get_data_for_tag(self, station_name, tag):
        """
        Returns the waveform and station data for the requested station and
        tag.

        :param station_name:
        :param tag:
        :return: tuple
        """
        contents = self.__file["Waveforms"][station_name].keys()
        traces = []
        for content in contents:
            if content.endswith("__%s" % tag):
                traces.append(self.get_waveform(content))

        st = obspy.Stream(traces=traces)
        inv = self.get_station(station_name)

        return st, inv

    def get_station(self, station_name):
        """
        Retrieves the specified StationXML as an obspy.station.Inventory object.
        """
        data = self.__file["Waveforms"][station_name]["StationXML"]
        inv = obspy.read_inventory(io.BytesIO(data.value.tostring()),
            format="stationxml")
        return inv

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
                str(trace.stats.starttime)
            station_group[data_name].attrs["sampling_rate"] = \
                str(trace.stats.sampling_rate)

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

    def process(self, process_function, output_filename):
        stations = self.__file["Waveforms"].keys()
        # Get all possible station and waveform tag combinations and let
        # each process read the data it needs.
        station_tags = []
        for station in stations:
            # Get the station and all possible tags.
            waveforms = self.__file["Waveforms"][station].keys()
            tags = set()
            for waveform in waveforms:
                if waveform == "StationXML":
                    continue
                tags.add(waveform.split("__")[-1])
            for tag in tags:
                station_tags.append((station, tag))

        # Check for MPI, if yes, dispatch to MPI worker, if not dispatch to
        # the multiprocessing handling.
        if _is_mpi_env():
            self._dispatch_processing_mpi(process_function, output_filename,
                                          station_tags)
        else:
            self._dispatch_processing_multiprocessing(
                process_function, output_filename, station_tags)

    def _dispatch_processing_mpi(self, process_function, output_filename,
                                 station_tags):
        import mpi4py

        self.comm = mpi4py.MPI.COMM_WORLD
        self.rank = self.comm.rank

        if self.rank == 0:
            self._dispatch_processing_mpi_master_node(process_function,
                                                     output_filename,
                                                     station_tags)
        else:
            self._dispatch_processing_mpi_worker_node(process_function,
                                                      output_filename)

    def _get_msg(self, source, tag):
        """
        Helper function to get a message if available, returns a
        ReceivedMessage instance in case a message is available, None
        otherwise.
        """
        tag = MSG_TAGS[tag]
        if not self.comm.Iprobe(source=source, tag=tag):
            return
        msg = ReceivedMessage(self.comm.recv(source=source, tag=tag))
        # XXX: No clue why this is necessary!
        while self.comm.Iprobe(source=source, tag=tag):
            self.comm.recv(source=source, tag=tag)
        if self.debug:
            pretty_receiver_log(source, self.rank, tag)
        return msg

    def _send_mpi(self, obj, dest, tag, blocking=True):
        tag = MSG_TAGS[tag]
        if blocking:
            self.comm.send(obj=obj, dest=dest, tag=tag)
        self.comm.isend(obj=obj, dest=dest, tag=tag)
        if self.debug:
            pretty_sender_log(dest, self.rank, tag)

    def _recv_mpi(self, source, tag):
        tag = MSG_TAGS[tag]
        msg = self.comm.recv(source=source, tag=tag)
        if self.debug:
            pretty_receiver_log(source, self.rank, tag)
        return msg

    def _dispatch_processing_mpi_master_node(self, process_function,
                                             output_filename, station_tags):
        """
        The master node. It distributes the jobs and takes care that
        metadata modifying actions are collective.
        """
        from mpi4py import MPI

        worker_nodes = range(1, self.comm.size)
        workers_requesting_write = []

        jobs = JobQueueHelper(jobs=station_tags,
                              worker_names=worker_nodes)

        __last_print = time.time()

        # Reactive event loop.
        while not jobs.all_done:
            time.sleep(0.01)

            # Informative output.
            if time.time() - __last_print > 2.0:
                print(jobs)
                __last_print = time.time()

            if len(workers_requesting_write) >= 0.5 * self.comm.size:
                if self.debug:
                    print("MASTER: initializing metadata synchronization.")
                for rank in worker_nodes:
                    self._send_mpi(None, rank, "MASTER_FORCES_WRITE")
                self._sync_metadata()
                workers_requesting_write[:] = []
                if self.debug:
                    print("MASTER: done with metadata synchronization.")
                continue

            # Retrieve any possible message and "dispatch" appropriately.
            status = MPI.Status()
            msg = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                 status=status)
            tag = MSG_TAGS[status.tag]
            source = status.source

            if self.debug:
                pretty_receiver_log(source, self.rank, status.tag)

            if tag == "WORKER_REQUESTS_ITEM":
                # Send poison pill if no more work is available. After
                # that the worker should not request any more jobs.
                if jobs.queue_empty:
                    self._send_mpi(POISON_PILL, source, "MASTER_SENDS_ITEM")
                else:
                    # And send a new station tag to process it.
                    station_tag = jobs.get_job_for_worker(source)
                    self._send_mpi(station_tag, source, "MASTER_SENDS_ITEM")

            elif tag == "WORKER_DONE_WITH_ITEM":
                station_tag, result = msg
                jobs.received_job_from_worker(station_tag, result, source)

            elif tag == "WORKER_REQUESTS_WRITE":
                workers_requesting_write.append(source)

            else:
                raise NotImplementedError

        # Shutdown workers.
        for rank in worker_nodes:
            self._send_mpi(None, rank, "ALL_DONE")

    def _dispatch_processing_mpi_worker_node(self, process_function,
                                             output_filename):
        """
        A worker node. It gets jobs, processes them and periodically waits
        until a collective metadata update operation has happened.
        """
        self.stream_buffer = StreamBuffer()

        worker_state = {
            "poison_pill_received": False,
            "waiting_for_write": False,
            "waiting_for_item": False
        }

        # Loop until the 'ALL_DONE' message has been sent.
        while not self._get_msg(0, "ALL_DONE"):
            time.sleep(0.01)

            # Check if master requested a write.
            if self._get_msg(0, "MASTER_FORCES_WRITE"):
                self._sync_metadata()
                for key, value in self.stream_buffer.items():
                    self._send_mpi((key, str(value)), 0,
                                   "WORKER_DONE_WITH_ITEM",
                                   blocking=False)
                self.stream_buffer.clear()
                worker_state["waiting_for_write"] = False

            if worker_state["waiting_for_write"]:
                continue

            if worker_state["poison_pill_received"]:
                continue

            if not worker_state["waiting_for_item"]:
                # Send message that the worker requires work.
                self._send_mpi(None, 0, "WORKER_REQUESTS_ITEM", blocking=False)
                worker_state["waiting_for_item"] = True
                continue

            msg = self._get_msg(0, "MASTER_SENDS_ITEM")
            if msg:
                station_tag = msg.data
                worker_state["waiting_for_item"] = False

                # If no more work to be done, store state and keep looping as
                # stuff still might require to be written.
                if station_tag == POISON_PILL:
                    if self.stream_buffer:
                        self._send_mpi(None, 0, "WORKER_REQUESTS_WRITE",
                                       blocking=False)
                    worker_state["poison_pill_received"] = True
                    continue

                # Otherwise process the data.
                stream, inv = self.get_data_for_tag(*station_tag)
                process_function(stream, inv)

                # Add stream to buffer.
                self.stream_buffer[station_tag] = stream

                # If the buffer is too large, request from the master to stop
                # the current execution.
                if self.stream_buffer.get_size() >= \
                                MAX_MEMORY_PER_WORKER_IN_MB * 1024 ** 2:
                    self._send_mpi(None, 0, "WORKER_REQUESTS_WRITE",
                                   blocking=False)
                    worker_state["waiting_for_write"] = True

    def _sync_metadata(self):

        if hasattr(self, "stream_buffer"):
            sendobj = self.stream_buffer.get_meta()
        else:
            sendobj = None

        data = self.comm.allgather(sendobj=sendobj)
        self.comm.barrier()

        # Make sure all remaining write requests are processed before
        # proceeding.
        if self.rank == 0:
            for rank in [1, 2, 3]:
                msg = self._get_msg(rank, "WORKER_REQUESTS_WRITE")
                if self.debug and msg:
                    print("MASTER: Ignoring write request by worker %i" % i)

        self.comm.barrier()

    def _dispatch_processing_multiprocessing(
            self, process_function, output_filename, station_tags):

        input_filename = self.__file.filename

        input_file_lock = multiprocessing.Lock()
        output_file_lock = multiprocessing.Lock()

        cpu_count = multiprocessing.cpu_count()

        # Create the input queue containing the jobs.
        input_queue = multiprocessing.JoinableQueue(
            maxsize=int(math.ceil(1.1 * (len(station_tags) + cpu_count))))

        for _i in station_tags:
            input_queue.put(_i)

        # Put some poison pills.
        for _ in xrange(cpu_count):
            input_queue.put(POISON_PILL)

        # Give a short time for the queues to play catch-up.
        time.sleep(0.1)

        # The output queue will collect the reports from the jobs.
        output_queue = multiprocessing.Queue()

        # Initialize the output file once.
        output_data_set = SDFDataSet(output_filename)

        class Process(multiprocessing.Process):
            def __init__(self, in_queue, out_queue, in_filename,
                         out_filename, in_lock, out_lock,
                         processing_function):
                super(Process, self).__init__()
                self.input_queue = in_queue
                self.output_queue = out_queue
                self.input_filename = in_filename
                self.output_filename = out_filename
                self.input_file_lock = in_lock
                self.output_file_lock = out_lock
                self.processing_function = processing_function

                with self.input_file_lock:
                    self.input_data_set = SDFDataSet(input_filename)

                with self.output_file_lock:
                    self.input_data_set = SDFDataSet(input_filename)

            def run(self):
                while True:
                    stationtag = self.input_queue.get(timeout=1)
                    if stationtag == POISON_PILL:
                        self.input_queue.task_done()
                        break

                    station, tag = stationtag
                    print station, tag

                    with self.input_file_lock:
                        stream, inv = \
                            self.input_data_set.get_data_for_tag(station,
                                                                 tag)
                    print "Processing..."
                    output_stream = self.processing_function(stream, inv)

                    self.input_queue.task_done()

        # Create n processes, with n being the number of available CPUs.
        processes = []
        for _ in xrange(cpu_count):
            processes.append(Process(input_queue, output_queue,
                                     input_filename, output_filename,
                                     input_file_lock, output_file_lock,
                                     process_function))

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        return


def apply_processing_multiprocessing(process_function, station, tag):
    """
    Applies the processing using the provided locks.
    """
    print len(input_data_set_container)
    # with input_lock:
    #     stream, inv = input_data_set.get_data_for_tag(station, tag)
    #
    # process_function = dill.loads(process_function)
    #
    # output_stream = process_function(stream, inv)
    #
    # with output_lock:
    #     output_data_set.add_stationxml(inv)
    #     output_data_set.add_waveform_file(output_stream, tag)

@property
def mpi(self):
    if hasattr(self, "__is_mpi"):
        return self.__is_mpi
    else:
        self.__is_mpi = self.__is_mpi_env()
    return self._is_mpi

def __is_mpi_env(self):
    """
    Returns True if the current environment is an MPI environment.
    """
    try:
        import mpi4py
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.size == 1 and mpi4py.MPI.COMM_WORLD.rank == 0:
        return False
    return True


class StreamBuffer(DictMixin):
    """
    Very simple key value store for obspy stream object with the additional
    ability to approximate the size of all stored stream objects.
    """
    def __init__(self):
        self.__streams = {}

    def __getitem__(self, key):
        return self.__streams[key]

    def __setitem__(self, key, value):
        if not isinstance(value, obspy.Stream):
            raise TypeError
        self.__streams[key] = value

    def __delitem__(self, key):
        del self.__streams[key]

    def keys(self):
        return self.__streams.keys()

    def get_size(self):
        """
        Try to approximate the size of all stores Stream object.
        """
        cum_size = 0
        for stream in self.__streams.itervalues():
            cum_size += sys.getsizeof(stream)
            for trace in stream:
                cum_size += sys.getsizeof(trace)
                cum_size += sys.getsizeof(trace.stats)
                cum_size += sys.getsizeof(trace.stats.__dict__)
                cum_size += sys.getsizeof(trace.data)
                cum_size += trace.data.nbytes
        # Add one percent buffer just in case.
        return cum_size * 1.01

    def get_meta(self):
        return "\n".join(str(_i) for _i in self.__streams.keys())


# Two objects describing a job and a worker.
class Job(object):
    __slots__ = "arguments", "result"

    def __init__(self, arguments, result=None):
        self.arguments = arguments
        self.result = result

    def __repr__(self):
        return "Job(arguments=%s, result=%s)" % (str(self.arguments),
                                                 str(self.result))

Worker = collections.namedtuple("Worker", ["jobs"])


class JobQueueHelper(object):
    """
    A simple helper class managing job distribution to workers.
    """
    def __init__(self, jobs, worker_names):
        """
        Init with a list of jobs and a list of workers.

        :type jobs: List of arguments distributed to the jobs.
        :param jobs: A list of jobs that will be distributed to the workers.
        :type: list of integers
        :param workers: A list of usually integers, each denoting a worker.
        """
        self._all_jobs = [Job(_i) for _i in jobs]
        self._in_queue = self._all_jobs[:]
        self._finished_jobs = []

        self._workers = {_i: Worker([]) for _i in worker_names}

    def get_job_for_worker(self, worker_name):
        """
        Get a job for a worker.

        :param worker_name: The name of the worker requesting work.
        """
        job = self._in_queue.pop(0)
        self._workers[worker_name].jobs.append(job)
        return job.arguments

    def received_job_from_worker(self, arguments, result, worker_name):
        """
        Call when a worker returned a job.

        :param arguments: The arguments the jobs was called with.
        :param result: The result of the job
        :param worker_name: The name of the worker.
        """
        # Find the correct job.
        job = [_i for _i in self._workers[worker_name].jobs
               if _i.arguments == arguments]
        assert len(job) == 1
        job = job[0]
        job.result = result

        self._workers[worker_name].jobs.remove(job)
        self._finished_jobs.append(job)

    def __str__(self):
        workers = "\n\t".join([
            "Worker %s: %i jobs" % (str(key), len(value.jobs))
            for key, value in self._workers.items()])

        return (
            "Jobs: In Queue: %i|Finished: %i|Total:%i\n"
            "\t%s\n" % (len(self._in_queue), len(self._finished_jobs),
                        len(self._all_jobs), workers))

    @property
    def queue_empty(self):
        return not bool(self._in_queue)

    @property
    def finished(self):
        return len(self._finished_jobs)

    @property
    def all_done(self):
        return len(self._all_jobs) == len(self._finished_jobs)


def pretty_sender_log(rank, destination, tag):
    import colorama
    prefix = colorama.Fore.RED + "sent to      " + colorama.Fore.RESET
    _pretty_log(prefix, destination, rank, tag)

def pretty_receiver_log(source, rank, tag):
    import colorama
    prefix = colorama.Fore.GREEN + "received from" + colorama.Fore.RESET
    _pretty_log(prefix, rank, source, tag)

def _pretty_log(prefix, first, second, tag):
    import colorama

    colors = (colorama.Back.WHITE + colorama.Fore.MAGENTA,
              colorama.Back.WHITE + colorama.Fore.BLUE,
              colorama.Back.WHITE + colorama.Fore.GREEN,
              colorama.Back.WHITE + colorama.Fore.YELLOW)

    tag_colors = (
        colorama.Fore.RED,
        colorama.Fore.GREEN,
        colorama.Fore.BLUE,
        colorama.Fore.YELLOW,
        colorama.Fore.MAGENTA,
    )

    tags = [i for i in MSG_TAGS.keys() if isinstance(i, basestring)]

    tag = MSG_TAGS[tag]
    tag = tag_colors[tags.index(tag) % len(tag_colors)] + tag + \
          colorama.Style.RESET_ALL

    first = colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET \
        if first == 0 else colors[first % len(colors)] + \
        ("WORKER %i" %  first) + colorama.Style.RESET_ALL
    second = colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET \
        if second == 0 else colors[second % len(colors)] + \
                           ("WORKER %i" %  second) + colorama.Style.RESET_ALL

    print("%s %s %s [%s]" % (first, prefix, second, tag))
