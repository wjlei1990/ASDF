from __future__ import absolute_import

import collections
import obspy
import sys
import time
from UserDict import DictMixin
import weakref

from .header import MSG_TAGS

# Tuple holding a the body of a received message.
ReceivedMessage = collections.namedtuple("ReceivedMessage", ["data"])
# Tuple denoting a single worker.
Worker = collections.namedtuple("Worker", ["active_jobs",
                                           "completed_jobs_count"])


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


class AuxiliaryDataContainer(object):
    def __init__(self, data, data_type, tag, parameters):
        self.data = data
        self.data_type = data_type
        self.tag = tag
        self.parameters = parameters

    def __str__(self):
        return (
            "Auxiliary Data of Type '{data_type}'\n"
            "\tTag: '{tag}'\n"
            "\tData shape: '{data_shape}', dtype: '{dtype}'\n"
            "\tParameters:\n\t\t{parameters}"
            .format(data_type=self.data_type, data_shape=self.data.shape,
                    dtype=self.data.dtype, tag=self.tag,
                    parameters="\n\t\t".join(["%s: %s" % (_i[0], _i[1])
                                              for _i in
                        sorted(self.parameters.items(), key=lambda x: x[0])])))


class AuxiliaryDataAccessor(object):
    """
    Helper class facilitating access to the actual waveforms and stations.
    """
    def __init__(self, auxiliary_data_type, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__auxiliary_data_type = auxiliary_data_type
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        return self.__data_set()._get_auxiliary_data(
            self.__auxiliary_data_type, item.replace("___", "."))

    def __dir__(self):
        __group = self.__data_set()._auxiliary_data_group[
            self.__auxiliary_data_type]
        return sorted([_i.replace(".", "___") for _i in __group.keys()])


class AuxiliaryDataGroupAccessor(object):
    """
    Helper class to facilitate access to the auxiliary data types.
    """
    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        __auxiliary_data_group = self.__data_set()._auxiliary_data_group
        if item not in __auxiliary_data_group:
            raise AttributeError
        return AuxiliaryDataAccessor(item, self.__data_set())

    def __dir__(self):
        __auxiliary_group = self.__data_set()._auxiliary_data_group
        return sorted(__auxiliary_group.keys())

    def __len__(self):
        return len(self.__dir__())


class StationAccessor(object):
    """
    Helper class to facilitate access to the waveforms and stations.
    """
    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        __waveforms = self.__data_set()._waveform_group
        if item.replace("_", ".") not in __waveforms:
            raise AttributeError
        return WaveformAccessor(item.replace("_", "."), self.__data_set())

    def __dir__(self):
        __waveforms = self.__data_set()._waveform_group
        return sorted(set(
            [_i.replace(".", "_") for _i in __waveforms.iterkeys()]))

    def __len__(self):
        return len(self.__dir__())


class WaveformAccessor(object):
    """
    Helper class facilitating access to the actual waveforms and stations.
    """
    def __init__(self, station_name, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__station_name = station_name
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        if item != "StationXML":
            __station = self.__data_set()._waveform_group[self.__station_name]
            keys = [_i for _i in __station.iterkeys()
                    if _i.endswith("__" + item)]
            traces = [self.__data_set()._get_waveform(_i) for _i in keys]
            return obspy.Stream(traces=traces)
        else:
            return self.__data_set()._get_station(self.__station_name)

    def __dir__(self):
        __station = self.__data_set()._waveform_group[self.__station_name]
        directory = []
        if "StationXML" in __station:
            directory.append("StationXML")
        directory.extend([_i.split("__")[-1]
                          for _i in __station.iterkeys()
                          if _i != "StationXML"])
        return sorted(set(directory))


def is_mpi_env():
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


# Two objects describing a job and a worker.
class Job(object):
    __slots__ = "arguments", "result"

    def __init__(self, arguments, result=None):
        self.arguments = arguments
        self.result = result

    def __repr__(self):
        return "Job(arguments=%s, result=%s)" % (str(self.arguments),
                                                 str(self.result))


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

        self._workers = {_i: Worker([], [0]) for _i in worker_names}

        self._starttime = time.time()

    def get_job_for_worker(self, worker_name):
        """
        Get a job for a worker.

        :param worker_name: The name of the worker requesting work.
        """
        job = self._in_queue.pop(0)
        self._workers[worker_name].active_jobs.append(job)
        return job.arguments

    def received_job_from_worker(self, arguments, result, worker_name):
        """
        Call when a worker returned a job.

        :param arguments: The arguments the jobs was called with.
        :param result: The result of the job
        :param worker_name: The name of the worker.
        """
        # Find the correct job.
        job = [_i for _i in self._workers[worker_name].active_jobs
               if _i.arguments == arguments]
        if len(job) == 0:
            msg = ("MASTER: Job %s from worker %i not found. All jobs: %s\n"
                  % (str(arguments), worker_name,
                     str(self._workers[worker_name].active_jobs)))
            raise ValueError(msg)
        if len(job) > 1:
            raise ValueError("WTF %i %s %s" % (worker_name, str(arguments),
                str(self._workers[worker_name].active_jobs)))
        job = job[0]
        job.result = result

        self._workers[worker_name].active_jobs.remove(job)
        self._workers[worker_name].completed_jobs_count[0] += 1
        self._finished_jobs.append(job)

    def __str__(self):
        workers = "\n\t".join([
            "Worker %s: %i active, %i completed jobs" %
            (str(key), len(value.active_jobs), value.completed_jobs_count[0])
            for key, value in self._workers.items()])

        return (
            "Jobs (running %.2f seconds): "
            "queued: %i | finished: %i | total: %i\n"
            "\t%s\n" % (time.time() - self._starttime, len(self._in_queue),
                        len(self._finished_jobs), len(self._all_jobs),
                        workers))

    @property
    def queue_empty(self):
        return not bool(self._in_queue)

    @property
    def finished(self):
        return len(self._finished_jobs)

    @property
    def all_done(self):
        return len(self._all_jobs) == len(self._finished_jobs)


def pretty_sender_log(rank, destination, tag, payload):
    import colorama
    prefix = colorama.Fore.RED + "sent to      " + colorama.Fore.RESET
    _pretty_log(prefix, destination, rank, tag, payload)


def pretty_receiver_log(source, rank, tag, payload):
    import colorama
    prefix = colorama.Fore.GREEN + "received from" + colorama.Fore.RESET
    _pretty_log(prefix, rank, source, tag, payload)


def _pretty_log(prefix, first, second, tag, payload):
    import colorama

    colors = (colorama.Back.WHITE + colorama.Fore.MAGENTA,
              colorama.Back.WHITE + colorama.Fore.BLUE,
              colorama.Back.WHITE + colorama.Fore.GREEN,
              colorama.Back.WHITE + colorama.Fore.YELLOW,
              colorama.Back.WHITE + colorama.Fore.BLACK,
              colorama.Back.WHITE + colorama.Fore.RED,
              colorama.Back.WHITE + colorama.Fore.CYAN)

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

    print("%s %s %s [%s] -- %s" % (first, prefix, second, tag, str(payload)))
