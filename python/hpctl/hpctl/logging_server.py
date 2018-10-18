from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves.queue import Empty
from six.moves import socketserver

import json
import pickle
import select
import struct
import socket
import atexit
import logging.handlers
from ctypes import c_bool
from multiprocessing import Process, Queue, Value
from baseline.utils import export as exporter
from hpctl.utils import Label


__all__ = []
export = exporter(__all__)


@export
class JSONSocketHandler(logging.handlers.SocketHandler):
    """Network logging handler that outputs JSON.

    :param label: str, The identifier to include in the log.

    The python network log format is a Big Endian Long that tells
    you how big the data is which is just serialized json.
    """
    def __init__(self, label=None, *args, **kwargs):
        super(JSONSocketHandler, self).__init__(*args, **kwargs)
        self.label = label
        if label is None:
            self.label = socket.gethostname()

    def makePickle(self, record):
        """Serialize the log record.

        :param record: logging.LogRecord, The log record.
        """
        d = dict(record.__dict__)
        # If there args then the log is a formatting string so we interpolate.
        # Otherwise we leave it there to get json encoded.
        if d['args']:
            d['msg'] = record.getMessage()
        d['args'] = None
        d['exc_info'] = None
        d.pop('message', None)
        # Inject a label into the log record we use to map a message to a job
        d['label'] = self.label
        data = json.dumps(d).encode('utf-8')
        length = struct.pack(">L", len(data))
        return length + data


class JSONStreamHandler(socketserver.StreamRequestHandler):
    """Handler to read the JSON network logging format.

    The python network log format is a Big Endian Long that tells
    you how big the data is which is just serialized json.
    """
    def handle(self):
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = json.loads(chunk.decode('utf-8'))
            self.server.queue.put(obj)


class LoggingServer(socketserver.ThreadingTCPServer):
    """Server to get logging messages.

    :param queue: queue.Queue, A queue to save incoming messages into.
    :param host: str, The hosts that are allowed to connect.
    :param port: int, The port to use.
    :param handler: socketserver.StreamRequestHandler, Class that is used to
        process an incoming request.
    :param timeout: int, The time to block on a socket.
    """
    allow_reuse_address = 1

    def __init__(
        self, queue,
        host='0.0.0.0', port=6006,
        handler=JSONStreamHandler,
        timeout=1
    ):
        self.queue = queue
        # socketserver is not a new style class and doesn't support super
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        # Shared memory sentinel to kill the server
        self.stop = Value(c_bool, False)
        self.timeout = timeout

    def serve(self):
        while not self.stop.value:
            rd, _, _ = select.select(
                [self.socket.fileno()], [], [],
                self.timeout
            )
            if rd:
                self.handle_request()


@export
class Logs(object):
    """Server wrapper than allows for access in a non blocking way.

    :param host: IPs that are allowed to connect.
    :param port: Port to listen on.
    :param handler: The class used to process a request.
    :param timeout: How long to wait when checking for new data.
    :param server_timeout: How long the server should wait for new data.
    """
    def __init__(
            self,
            host='', port=6006,
            handler=JSONStreamHandler,
            timeout=1, server_timeout=1
    ):
        self.timeout = timeout
        self.q = Queue()
        self.server = LoggingServer(
            self.q,
            host=host, port=port,
            handler=handler, timeout=server_timeout
        )
        self.server_process = Process(target=self.server.serve)
        atexit.register(self.stop)
        self.server_process.start()


    def get(self):
        """Get logs.

        Returns:
            Tuple[id, data] if data is available, else None, None
        """
        try:
            data = self.q.get(timeout=self.timeout)
        except Empty:
            data = None
        if data is not None:
            label = Label.parse(data['label'])
            return label, data['msg']
        return None, None

    def stop(self):
        """Stop the server process and block until done."""
        self.server.stop.value = True
        self.server_process.join()

    @classmethod
    def create(cls, hpctl_logs):
        port = hpctl_logs['port']
        return cls(port=port)


class DummyLogs(object):
    def __init__(self, *args, **kwargs):
        super(DummyLogs, self).__init__()

    def get(self):
        return None, None

    def stop(self):
        pass

    @classmethod
    def create(cls, hpctl_logs):
        return cls()


def get_log_server(log_config):
    kind = log_config.pop('type', 'real')
    if kind == 'remote':
        return DummyLogs.create(log_config)
    return Logs.create(log_config)
