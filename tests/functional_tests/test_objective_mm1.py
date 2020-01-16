import numpy as np
import pytest
from numpy.random.mtrand import exponential

from pydesim import simulate, Model, Statistic, Intervals, Trace


class QueueingSystem(Model):
    """QueueingSystem model is the top-level model for this test.

    This model consists of four children:
    - `queue`: a model representing the packets queue (`Queue`)
    - `source`: a model representing the packets source (`Source`)
    - `server`: a model representing the packets server (`Server`)
    - `sink`: a model which collects served packets (`Sink`)
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.children['queue'] = Queue(sim, sim.params.capacity)
        self.children['source'] = Source(sim, sim.params.arrival_mean)
        self.children['server'] = Server(sim, sim.params.service_mean)
        self.children['sink'] = Sink(sim)

        # Building connections:
        self.source.connections['queue'] = self.queue
        self.queue.connections['server'] = self.server
        self.server.connections['queue'] = self.queue
        self.server.connections['sink'] = self.sink

        # Statistics:
        self.system_size_trace = Trace()
        self.system_size_trace.record(self.sim.stime, 0)
    
    @property
    def queue(self):
        return self.children['queue']
    
    @property
    def source(self):
        return self.children['source']
    
    @property
    def server(self):
        return self.children['server']
    
    @property
    def sink(self):
        return self.children['sink']
    
    @property
    def system_size(self):
        return self.queue.size + (1 if self.server.busy else 0)
    
    def update_system_size(self):
        self.system_size_trace.record(self.sim.stime, self.system_size)


class Queue(Model):
    """Queue module represents the packets queue, stores only current size.

    Connections: server

    Methods and properties:
    - push(): increase the queue size
    - pop(): decrease the queue size
    - size: get current queue size

    Statistics:
    -  size_trace: Trace, holding the history of the queue size updates
    """
    def __init__(self, sim, capacity):
        super().__init__(sim)
        self.__capacity = capacity
        self.__size = 0
        # Statistics:
        self.size_trace = Trace()
        self.size_trace.record(self.sim.stime, 0)
    
    @property
    def capacity(self):
        return self.__capacity
    
    @property
    def size(self):
        return self.__size
    
    def push(self):
        server = self.connections['server'].module
        if self.__size == 0 and not server.busy:
            server.start_service()
        elif self.capacity < 0 or self.__size < self.capacity:
            self.__size += 1
            self.size_trace.record(self.sim.stime, self.__size)
        self.parent.update_system_size()
    
    def pop(self):
        assert self.__size > 0
        self.__size -= 1
        self.size_trace.record(self.sim.stime, self.__size)
    
    def __str__(self):
        return f'Queue({self.size})'


class Source(Model):
    """Source module represents the traffic source with exponential intervals.

    Connections: queue

    Handlers:
    - on_timeout(): called upon next arrival timeout

    Statistics:
    - intervals: `Intervals`, stores a set of inter-arrival intervals

    Parent: `QueueingSystem`
    """
    def __init__(self, sim, arrival_mean):
        super().__init__(sim)
        self.__arrival_mean = arrival_mean
        # Statistics:
        self.intervals = Intervals()
        # Initialize:
        self._schedule_next_arrival()

    @property
    def arrival_mean(self):
        return self.__arrival_mean

    def on_timeout(self):
        queue = self.connections['queue'].module
        queue.push()
        self._schedule_next_arrival()
    
    def _schedule_next_arrival(self):
        self.intervals.record(self.sim.stime)
        self.sim.schedule(exponential(self.arrival_mean), self.on_timeout)


class Server(Model):
    """Server module represents a packet server with exponential service time.

    Connections: queue, sink

    Handlers:
    - on_service_end(): called upon service timeout

    Methods:
    - start_service(): start new packet service; generate error if busy.

    Statistics:
    - delays: `Statistic`, stores a set of service intervals
    - busy_trace: `Trace`, stores a vector of server busy status

    Parent: `QueueingSystem`
    """
    def __init__(self, sim, service_mean):
        super().__init__(sim)
        self.__service_mean = service_mean
        self.__busy = False
        # Statistics:
        self.delays = Statistic()
        self.busy_trace = Trace()
        self.busy_trace.record(self.sim.stime, 0)
    
    @property
    def service_mean(self):
        return self.__service_mean
    
    @property
    def busy(self):
        return self.__busy
    
    def on_service_end(self):
        assert self.__busy
        queue = self.connections['queue'].module
        self.__busy = False
        self.busy_trace.record(self.sim.stime, 0)
        if queue.size > 0:
            queue.pop()
            self.start_service()
        self.connections['sink'].module.receive_packet()
        self.parent.update_system_size()

    def start_service(self):
        assert not self.__busy
        delay = exponential(self.service_mean)
        self.sim.schedule(delay, self.on_service_end)
        self.delays.append(delay)
        self.__busy = True
        self.busy_trace.record(self.sim.stime, 1)


class Sink(Model):
    """Sink module represents the traffic sink and counts arrived packets.

    Methods:
    - receive_packet(): called when the server finishes serving packet.
    """
    def __init__(self, sim):
        super().__init__(sim)
        # Statistics:
        self.departures = Intervals()
        self.departures.record(self.sim.stime)
    
    def receive_packet(self):
        self.departures.record(self.sim.stime)


@pytest.mark.parametrize('mean_arrival,mean_service', [(2., 1.), (5., 2.)])
def test_objective_mm1_model(mean_arrival, mean_service):
    ret = simulate(QueueingSystem, stime_limit=4000, params={
        'arrival_mean': mean_arrival,
        'service_mean': mean_service,
        'capacity': -1,
    })

    busy_rate = ret.data.server.busy_trace.timeavg()
    system_size = ret.data.system_size_trace.timeavg()
    est_arrival_mean = ret.data.source.intervals.statistic().mean()
    est_departure_mean = ret.data.sink.departures.statistic().mean()
    est_service_mean = ret.data.server.delays.mean()

    rho = mean_service / mean_arrival

    assert np.allclose(est_service_mean, mean_service, rtol=0.2)
    assert np.allclose(busy_rate, rho, rtol=0.2)
    assert np.allclose(system_size, rho / (1 - rho), atol=0.05, rtol=0.2)
    assert np.allclose(est_arrival_mean, mean_arrival, rtol=0.2)
    assert np.allclose(est_departure_mean, mean_arrival, rtol=0.2)
