#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prototype implementation for a new file format using Python, ObsPy, and HDF5.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import

import copy
import collections
import h5py
import io
import multiprocessing
import math
import itertools
import numpy as np
import obspy
import os
import warnings
import time

from header import SDFException, SDFWarnings, COMPRESSIONS, FORMAT_NAME, \
    FORMAT_VERSION, MSG_TAGS, MAX_MEMORY_PER_WORKER_IN_MB, POISON_PILL
from utils import is_mpi_env, StationAccessor, sizeof_fmt, ReceivedMessage,\
    pretty_receiver_log, pretty_sender_log, JobQueueHelper, StreamBuffer


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
        :type debug: bool
        :param debug: If True, print debug messages. Defaults to False.
        """
        self.debug = debug
        if compression not in COMPRESSIONS:
            msg = "Unknown compressions '%s'. Available compressions: \n\t%s" \
                % (compression, "\n\t".join(sorted(
                [str(i) for i in COMPRESSIONS.keys()])))
            raise Exception(msg)
        self.__compression = COMPRESSIONS[compression]

        # Turn off compression for parallel I/O.
        if self.__compression[0] and self.mpi:
            msg = "Compression will be disabled as parallel HDF5 does not " \
                  "support compression"
            warnings.warn(msg)
            self.__compression = COMPRESSIONS[None]

        # Open file or take an already open HDF5 file object.
        if not self.mpi:
            if isinstance(file_object, h5py.File):
                self.__file = file_object
                if self.__file.mode != "r+":
                    raise ValueError("The file pointer must have mode 'r+'.")
            else:
                self.__file = h5py.File(file_object, "a")
        else:
            if isinstance(file_object, h5py.File):
                self.__file = file_object
                if self.__file.mode != "r+" or self.__file.driver != "mpio":
                    raise ValueError("A file pointer passed with MPI enabled "
                                     "must have mode 'r+' and the 'mpio' "
                                     "driver")
            else:
                self.__file = h5py.File(file_object, "a", driver="mpio",
                                        comm=self.mpi.comm)

        if "file_format" in self.__file.attrs:
            if self.__file.attrs["file_format"] != FORMAT_NAME:
                msg = "Not a '%s' file." % FORMAT_NAME
                raise SDFException(msg)
            if "file_format_version" not in self.__file.attrs:
                msg = ("No file format version given for file '%s'. The "
                       "program will continue but the result is undefined." %
                       self.__file.filename)
                warnings.warn(msg, SDFWarnings)
            elif self.__file.attrs["file_format_version"] != FORMAT_VERSION:
                msg = ("The file '%s' has version number '%s'. The reader "
                       "expects version '%s'. The program will continue but "
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

        # Force collective init if run in an MPI environment.
        if self.mpi:
            self.mpi.comm.barrier()

    def __del__(self):
        """
        Attempts to close the HDF5 file.
        """
        try:
            self.__file.close()
        # Value Error is raised if the file has already been closed.
        except ValueError:
            pass

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
        Retrieves the specified StationXML as an obspy.station.Inventory
        object.
        """
        data = self.__file["Waveforms"][station_name]["StationXML"]
        inv = obspy.read_inventory(io.BytesIO(data.value.tostring()),
            format="stationxml")
        return inv

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
            # Complicated multi-step process but it enables one to use
            # parallel I/O with the same functions.
            info = self._add_trace_get_collective_information(trace, tag)
            if info is None:
                continue
            self._add_trace_write_collective_information(info)
            self._add_trace_write_independent_information(info, trace)

    def _add_trace_write_independent_information(self, info, trace):
        """
        Writes the independent part of a trace to the file.

        :param info:
        :param trace:
        :return:
        """
        self.__waveforms[info["data_name"]][:] = trace.data

    def _add_trace_write_collective_information(self, info):
        """
        Writes the collective part of a trace to the file.

        :param info:
        :return:
        """
        station_name = info["station_name"]
        if not station_name in self.__waveforms:
            self.__waveforms.create_group(station_name)
        group = self.__waveforms[station_name]

        ds = group.create_dataset(**info["dataset_creation_params"])
        for key, value in info["dataset_attrs"].items():
            ds.attrs[key] = value

    def _add_trace_get_collective_information(self, trace, tag):
        """
        The information required for the collective part of adding a trace.

        This will extract the group name, the parameters of the dataset to
        be created, and the attributes of the dataset.

        :param trace: The trace to add.
        :param tag: The tag of the trace.
        """
        station_name = "%s.%s" % (trace.stats.network, trace.stats.station)
        # Generate the name of the data within its station folder.
        data_name = "{net}.{sta}.{loc}.{cha}__{start}__{end}__{tag}".format(
            net=trace.stats.network,
            sta=trace.stats.station,
            loc=trace.stats.location,
            cha=trace.stats.channel,
            start=trace.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"),
            end=trace.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S"),
            tag=tag)

        group_name = "%s/%s" % (station_name, data_name)
        if group_name in self.__waveforms:
            msg = "Data '%s' already exists in file. Will not be added!" % \
                  group_name
            warnings.warn(msg, SDFWarnings)
            return

        # XXX: Figure out why this is necessary. It should work according to
        # the specs.
        if self.mpi:
            fletcher32 = False
        else:
            fletcher32 = True

        return {
            "station_name": station_name,
            "data_name": group_name,
            "dataset_creation_params": {
                "name": data_name,
                "shape": (trace.stats.npts,),
                "dtype": trace.data.dtype,
                "compression": self.__compression[0],
                "compression_opts": self.__compression[1],
                "fletcher32": fletcher32,
                "maxshape": (None,)
            },
            "dataset_attrs": {
                "starttime": str(trace.stats.starttime),
                "sampling_rate": str(trace.stats.sampling_rate)
            }
        }


    def add_obspy_trace(self, trace, tag):
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
            return
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
        stations = sorted(self.__file["Waveforms"].keys())
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

        assert len(station_tags) == len(set(station_tags))

        # Check for MPI, if yes, dispatch to MPI worker, if not dispatch to
        # the multiprocessing handling.
        if self.mpi:
            self._dispatch_processing_mpi(process_function, output_filename,
                                          station_tags)
        else:
            self._dispatch_processing_multiprocessing(
                process_function, output_filename, station_tags)

    def _dispatch_processing_mpi(self, process_function, output_filename,
                                 station_tags):

        output_dataset = SDFDataSet(output_filename, debug=self.debug)

        if self.mpi.rank == 0:
            self._dispatch_processing_mpi_master_node(process_function,
                                                      output_dataset,
                                                      station_tags)
        else:
            self._dispatch_processing_mpi_worker_node(process_function,
                                                      output_dataset)

    def _dispatch_processing_mpi_master_node(self, process_function,
                                             output_dataset, station_tags):
        """
        The master node. It distributes the jobs and takes care that
        metadata modifying actions are collective.
        """
        from mpi4py import MPI

        worker_nodes = range(1, self.mpi.comm.size)
        workers_requesting_write = []

        jobs = JobQueueHelper(jobs=station_tags[:9],
                              worker_names=worker_nodes)

        __last_print = time.time()

        # Reactive event loop.
        while not jobs.all_done:
            time.sleep(0.01)

            # Informative output.
            if time.time() - __last_print > 2.0:
                print(jobs)
                __last_print = time.time()

            if len(workers_requesting_write) >= 0.5 * self.mpi.comm.size:
                if self.debug:
                    print("MASTER: initializing metadata synchronization.")

                # Send force write msgs to all workers and block until all
                # have been sent. Don't use blocking send cause then one
                # will have to wait each time anew and not just once for each.
                # The message will ready each worker for a collective
                # operation once its current operation is ready.
                requests = [self._send_mpi(None, rank, "MASTER_FORCES_WRITE",
                                           blocking=False)
                            for rank in worker_nodes]
                self.mpi.MPI.Request.waitall(requests)

                self._sync_metadata(output_dataset)

                # Reset workers requesting a write.
                workers_requesting_write[:] = []
                if self.debug:
                    print("MASTER: done with metadata synchronization.")
                continue

            # Retrieve any possible message and "dispatch" appropriately.
            status = MPI.Status()
            msg = self.mpi.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                 status=status)
            tag = MSG_TAGS[status.tag]
            source = status.source

            if self.debug:
                pretty_receiver_log(source, self.mpi.rank, status.tag, msg)

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

        self.mpi.comm.barrier()
        print(jobs)

    def _dispatch_processing_mpi_worker_node(self, process_function,
                                             output_dataset):
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
                self._sync_metadata(output_dataset)
                for key, value in self.stream_buffer.items():
                    tag = key[1]
                    for trace in value:
                        output_dataset.\
                            _add_trace_write_independent_information(
                                trace.stats.__info, trace)
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

        self.mpi.comm.barrier()

    def _sync_metadata(self, output_dataset):
        """
        Method responsible for synchronizing metadata across all processes
        in the HDF5 file. All metadata changing operations must be collective.
        """
        if hasattr(self, "stream_buffer"):
            sendobj = []
            for key, stream in self.stream_buffer.items():
                tag = key[1]
                for trace in stream:
                    info = \
                        output_dataset._add_trace_get_collective_information(
                            trace, tag)
                    trace.stats.__info = info
                    sendobj.append(info)
        else:
            sendobj = [None]

        data = self.mpi.comm.allgather(sendobj=sendobj)
        # Chain and remove None.
        trace_info = itertools.ifilter(lambda x: x is not None,
                                       itertools.chain.from_iterable(data))
        # Write collective part.
        for info in trace_info:
            output_dataset._add_trace_write_collective_information(info)

        # Make sure all remaining write requests are processed before
        # proceeding.
        if self.mpi.rank == 0:
            for rank in [1, 2, 3]:
                msg = self._get_msg(rank, "WORKER_REQUESTS_WRITE")
                if self.debug and msg:
                    print("MASTER: Ignoring write request by worker %i" %
                          rank)

        self.mpi.comm.barrier()

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

    @property
    def mpi(self):
        if hasattr(self, "__is_mpi"):
            return self.__is_mpi
        else:
            self.__is_mpi = is_mpi_env()

        # If it actually is an mpi environment, set the communicator and the
        # rank.
        if self.__is_mpi:
            import mpi4py

            # Set mpi tuple to easy class wide access.
            mpi_ns = collections.namedtuple("mpi_ns", ["comm", "rank",
                                                       "size", "MPI"])
            comm = mpi4py.MPI.COMM_WORLD
            self.__is_mpi = mpi_ns(comm=comm, rank=comm.rank,
                                   size=comm.size, MPI=mpi4py.MPI)

        return self.__is_mpi

    def _get_msg(self, source, tag):
        """
        Helper method to get a message if available, returns a
        ReceivedMessage instance in case a message is available, None
        otherwise.
        """
        tag = MSG_TAGS[tag]
        if not self.mpi.comm.Iprobe(source=source, tag=tag):
            return
        msg = ReceivedMessage(self.mpi.comm.recv(source=source, tag=tag))
        if self.debug:
            pretty_receiver_log(source, self.mpi.rank, tag, msg.data)
        return msg

    def _send_mpi(self, obj, dest, tag, blocking=True):
        """
        Helper method to send a message via MPI.
        """
        tag = MSG_TAGS[tag]
        if blocking:
            value = self.mpi.comm.send(obj=obj, dest=dest, tag=tag)
        else:
            value = self.mpi.comm.isend(obj=obj, dest=dest, tag=tag)
        if self.debug:
            pretty_sender_log(dest, self.mpi.rank, tag, obj)
        return value

    def _recv_mpi(self, source, tag):
        """
        Helper method to receive a message via MPI.
        """
        tag = MSG_TAGS[tag]
        msg = self.mpi.comm.recv(source=source, tag=tag)
        if self.debug:
            pretty_receiver_log(source, self.mpi.rank, tag, msg)
        return msg

