import numpy as np
import pytest
from numpy.random.mtrand import exponential

from pydesim import simulate, Trace, Intervals, Statistic


@pytest.mark.parametrize('mean_arrival,mean_service', [(2., 1.), (5., 2.)])
def test_mm1(mean_arrival, mean_service):
    """Validate estimation of M/M/1 system parameters using simulate().
    """
    def arrive(sim):
        if sim.data.server_busy:
            sim.data.queue_size += 1
        else:
            interval = exponential(sim.params.service_mean)
            sim.schedule(interval, depart)
            sim.data.server_busy = True

            # Write statistics about server status and service interval:
            sim.data.server_busy_trace.record(sim.stime, 1)
            sim.data.service_intervals.append(interval)

        sim.schedule(exponential(sim.params.arrival_mean), arrive)

        # Write statistics about system size and arrival timestamps:
        sim.data.system_size_trace.record(sim.stime, sim.data.system_size)
        sim.data.arrivals.record(sim.stime)

    def depart(sim):
        assert sim.data.system_size > 0
        if sim.data.queue_size > 0:
            sim.data.queue_size -= 1
            interval = exponential(sim.params.service_mean)
            sim.schedule(interval, depart)

            # Write statistics about service interval:
            sim.data.service_intervals.append(interval)
        else:
            sim.data.server_busy = False

            # Write statistics about server status update:
            sim.data.server_busy_trace.record(sim.stime, 0)

        # Write statistics about system size and departure timestamps:
        sim.data.system_size_trace.record(sim.stime, sim.data.system_size)
        sim.data.departures.record(sim.stime)

    def init(sim):
        sim.schedule(exponential(sim.params.arrival_mean), arrive)
        # Record initial points in statistics:
        sim.data.system_size_trace.record(0, 0)
        sim.data.server_busy_trace.record(0, 0)
        sim.data.arrivals.record(0)
        sim.data.departures.record(0)

    class ModelData:
        def __init__(self, arrival_mean, service_mean):
            # Parameters:
            self.arrival_mean = arrival_mean
            self.service_mean = service_mean

            # System state:
            self.queue_size = 0
            self.server_busy = False

            # Statistics:
            self.system_size_trace = Trace()
            self.server_busy_trace = Trace()
            self.service_intervals = Statistic()
            self.arrivals = Intervals()
            self.departures = Intervals()

        @property
        def system_size(self):
            return self.queue_size + int(self.server_busy)

    ret = simulate(ModelData, init=init, stime_limit=2000, params={
        'arrival_mean': mean_arrival,
        'service_mean': mean_service,
    })

    busy_rate = ret.data.server_busy_trace.timeavg()
    system_size = ret.data.system_size_trace.timeavg()
    est_arrival_mean = ret.data.arrivals.statistic().mean()
    est_departure_mean = ret.data.departures.statistic().mean()
    est_service_mean = ret.data.service_intervals.mean()

    rho = mean_service / mean_arrival

    assert np.allclose(est_service_mean, mean_service, rtol=0.1)
    assert np.allclose(busy_rate, rho, atol=0.05, rtol=0.1)
    assert np.allclose(system_size, rho / (1 - rho), atol=0.05, rtol=0.2)
    assert np.allclose(est_arrival_mean, mean_arrival, rtol=0.1)
    assert np.allclose(est_departure_mean, mean_arrival, rtol=0.1)
