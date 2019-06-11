import heapq
import itertools
from enum import Enum
from functools import total_ordering
import colorama


class HandlersDict:
    def __init__(self, handlers):
        self.__handlers = {}
        if handlers:
            self.__handlers.update(handlers)

    def __getitem__(self, item):
        return self.__handlers[item]

    def __setitem__(self, key, value):
        self.__handlers[key] = value

    def __getattr__(self, item):
        return self.__handlers[item]

    def get(self, item):
        return self.__handlers[item]


class _ParamsDict:
    def __init__(self, kwargs):
        self.__kwargs = {}
        self.__kwargs.update(kwargs if kwargs else {})

    def __getitem__(self, item):
        return self.__kwargs[item]

    def __getattr__(self, item):
        return self.__kwargs[item]

    def as_dict(self):
        d = {}
        d.update(self.__kwargs)
        return d


@total_ordering
class _Event:
    def __init__(self, id, stime, fn, args, kwargs):
        self.__id, self.__stime, self.__fn, self.__args, self.__kwargs = \
            id, stime, fn, args, kwargs
        self.__removed = False

    @property
    def id(self):
        return self.__id

    @property
    def stime(self):
        return self.__stime

    @property
    def fn(self):
        return self.__fn

    @property
    def args(self):
        return self.__args

    @property
    def kwargs(self):
        return self.__kwargs

    @property
    def removed(self):
        return self.__removed

    def remove(self):
        self.__removed = True

    def __eq__(self, other):
        return self.__stime == other.stime and self.__id == other.id

    def __lt__(self, other):
        return self.__stime < other.stime or (
                self.__stime == other.stime and self.__id < other.id)


class Kernel:
    def __init__(self):
        self.__queue = []
        self.__stime = 0
        self.__evids = {}
        self.__next_evid = itertools.count()
        self.__num_events = 0
        self.__queue_size = 0
        self.__stop_predicates = []

    @property
    def stime(self):
        return self.__stime

    @property
    def empty(self):
        return self.__queue_size == 0

    @property
    def num_events(self):
        return self.__num_events

    def add_event(self, delay, handler=None, args=(), kwargs=None):
        if delay < 0:
            raise ValueError('negative delay disallowed')
        kwargs = {} if kwargs is None else kwargs
        event = _Event(
            next(self.__next_evid), self.stime + delay, handler, args, kwargs)
        self.__evids[event.id] = event
        heapq.heappush(self.__queue, event)
        self.__queue_size += 1
        return event.id

    def remove_event(self, evid):
        event = self.__evids.get(evid, None)
        if event:
            del self.__evids[evid]
            if not event.removed:
                event.remove()
                self.__queue_size -= 1
                return event
        return None

    def _next_event(self):
        while self.__queue:
            event = heapq.heappop(self.__queue)
            if not event.removed:
                # Update time:
                assert event.stime >= self.__stime
                self.__stime = event.stime

                # Remove event from the EventID table and reduce queue size:
                del self.__evids[event.id]
                self.__queue_size -= 1

                return event
        raise KeyError('pop from empty queue')

    def _test_stop(self):
        return any(pred(self) for pred in self.__stop_predicates)

    def setup(self, stime_limit=None):
        if stime_limit is not None and stime_limit > 0:
            self.__stop_predicates.append(
                lambda kern: stime_limit < kern.stime
            )

    def run(self, sim, init, fin):
        if hasattr(sim.data, 'initialize'):
            sim.data.initialize(sim)
        if init:
            init(sim)

        while not self.empty:
            event = self._next_event()
            if not self._test_stop():
                if event.fn:
                    if hasattr(event.fn, '__self__'):
                        sim.logger.trace(
                            f'** calling {event.fn.__name__}()',
                            src=event.fn.__self__
                        )
                        event.fn(*event.args, **event.kwargs)
                    else:
                        sim.logger.trace(
                            f'** {event.fn.__name__}()',
                            src='kernel'
                        )
                        event.fn(sim, *event.args, **event.kwargs)
                    self.__num_events += 1
            else:
                break

        if fin:
            fin(sim)


class Logger:
    class Level(Enum):
        TRACE = 0
        DEBUG = 1
        INFO = 2
        WARNING = 3
        ERROR = 4

    def __init__(self, kernel):
        self.level = Logger.Level.INFO
        self.__kernel = kernel

    @property
    def kernel(self):
        return self.__kernel

    def write(self, level, msg, src=''):
        fs_bright = colorama.Style.BRIGHT
        fs_normal = colorama.Style.NORMAL
        fs_dim = colorama.Style.DIM
        fs_reset = colorama.Style.RESET_ALL + colorama.Fore.RESET
        time_color = colorama.Fore.LIGHTCYAN_EX

        if level.value >= self.level.value:
            lc = Logger.level2font(level)
            src_str = (fs_bright + f'({src}) ' + fs_reset) if src else ''
            level_str = fs_bright + lc + f'[{level.name:7s}]'
            time_str = fs_dim + time_color + f'{self.kernel.stime:014.9f}'
            msg_str = fs_normal + lc + msg
            print(f'{level_str} {time_str} {src_str}{msg_str}' + fs_reset)

    def trace(self, msg, src=''):
        self.write(Logger.Level.TRACE, msg, src)

    def debug(self, msg, src=''):
        self.write(Logger.Level.DEBUG, msg, src)

    def info(self, msg, src=''):
        self.write(Logger.Level.INFO, msg, src)

    def warning(self, msg, src=''):
        self.write(Logger.Level.WARNING, msg, src)

    def error(self, msg, src=''):
        self.write(Logger.Level.ERROR, msg, src)

    @staticmethod
    def level2font(level):
        if level is Logger.Level.TRACE:
            return colorama.Fore.LIGHTBLACK_EX
        if level is Logger.Level.DEBUG:
            return colorama.Fore.WHITE
        if level is Logger.Level.INFO:
            return colorama.Fore.MAGENTA
        if level is Logger.Level.WARNING:
            return colorama.Fore.YELLOW
        if level is Logger.Level.ERROR:
            return colorama.Fore.RED


class Simulator:
    def __init__(self, kernel, protodata, handlers, params=None, loglevel=None):
        params = {} if params is None else params
        self.__handlers = HandlersDict(handlers)
        self.__kernel = kernel
        self.__params = _ParamsDict(params)
        self.__logger = Logger(kernel)
        if loglevel is not None:
            self.__logger.level = loglevel
        # Creating model data:
        if isinstance(protodata, type):
            # If protodata is a class, then we create an instance of it with
            # using params as kwargs. The rules are:
            # - if `protodata` is subclassed from `Model`, we instantiate it
            #   like `CustomModel(sim=self)`;
            # - if `protodata` is is not subclassed from `Model`, but have
            #   special `create()` method, we call it with **params, but
            #   without passing simulator;
            # - in other cases, we simply call constructor with **params.
            if issubclass(protodata, Model):
                self.__data = protodata(self)
            elif hasattr(protodata, 'create'):
                self.__data = getattr(protodata, 'create')(**params)
            else:
                self.__data = protodata(**params)
        else:
            # If protodata is not a class, we use it as it is.
            self.__data = protodata

    @property
    def stime(self):
        return self.__kernel.stime

    @property
    def num_events(self):
        return self.__kernel.num_events

    def schedule(self, delay, handler=None, args=(), kwargs=None):
        return self.__kernel.add_event(delay, handler, args, kwargs)

    def cancel(self, evid):
        self.__kernel.remove_event(evid)

    @property
    def params(self):
        return self.__params

    @property
    def data(self):
        return self.__data

    @property
    def handlers(self):
        return self.__handlers

    @property
    def logger(self):
        return self.__logger


def simulate(data, init=None, fin=None, handlers=None, params=None,
             stime_limit=None, loglevel=Logger.Level.INFO):
    stime_limit = stime_limit if stime_limit is not None else 0

    if isinstance(params, list):
        results = []
        for a_params in params:
            kernel = Kernel()
            sim = Simulator(kernel, data, handlers, a_params, loglevel)
            kernel.setup(stime_limit=stime_limit)
            kernel.run(sim, init=init, fin=fin)
            results.append(sim)
        return results

    kernel = Kernel()
    sim = Simulator(kernel, data, handlers, params, loglevel)
    kernel.setup(stime_limit=stime_limit)
    kernel.run(sim, init=init, fin=fin)
    return sim


class _ModulesManager:
    def __init__(self, container):
        assert isinstance(container, dict)
        self.__container = container

    def add(self, name, module):
        if name in self.__container:
            del self.__container[name]
        self.__container[name] = module

    def update(self, kwargs):
        for name in kwargs:
            self.remove(name)
        self.__container.update(kwargs)

    def remove(self, name):
        try:
            del self.__container[name]
        except KeyError:
            pass

    def get(self, name):
        return self.__container[name]

    def all(self):
        return dict(self.__container)

    def modules(self):
        modules = []

        def add(val):
            if isinstance(val, Model):
                modules.append(val)
            else:
                for v in val:
                    add(v)

        for value in self.__container.values():
            add(value)

        return modules


class Model:
    def __init__(self, sim, parent=None, *args, **kwargs):
        self.__sim = sim
        self.__parent = parent
        self.__children = {}
        self.__modules = {}
        self.__children_manager = _ModulesManager(self.__children)
        self.__modules_manager = _ModulesManager(self.__modules)

    @property
    def sim(self):
        return self.__sim

    @sim.setter
    def sim(self, sim):
        self.__sim = sim

    @property
    def children(self):
        return self.__children_manager

    @property
    def connections(self):
        return self.__modules_manager

    @property
    def parent(self):
        return self.__parent
