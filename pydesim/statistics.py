import numpy as np


class Statistic:
    def __init__(self, data=None):
        if data is not None:
            self._data = list(data)
        else:
            self._data = []

    def append(self, value):
        self._data.append(value)

    def extend(self, data):
        self._data.extend(data)

    def mean(self):
        if self.empty:
            raise ValueError('no data')
        return self.asarray().mean()

    def std(self):
        if self.empty:
            raise ValueError('no data')
        return self.asarray().std()

    def var(self):
        if self.empty:
            raise ValueError('no data')
        return self.asarray().var()

    def moment(self, k):
        n = len(self)
        if n == 0:
            raise ValueError('no data')
        if np.abs(np.round(k) - k) > 0 or k <= 0:
            raise ValueError('positive integer expected')
        return sum((x ** k for x in self._data)) / n

    def lag(self, k):
        n = len(self)
        if n == 0:
            raise ValueError('no data')
        if np.abs(np.round(k) - k) > 0 or k < 0:
            raise ValueError('non-negative integer expected')
        if n <= k:
            raise ValueError('statistic has too few samples')
        ar = self.asarray()
        if k == 0:
            return 1
        return np.corrcoef(ar[k:], ar[:-k])[0, 1]

    def __len__(self):
        return len(self._data)

    @property
    def empty(self):
        return len(self._data) == 0

    def as_list(self):
        return list(self._data)

    def as_tuple(self):
        return tuple(self._data)

    def asarray(self):
        return np.asarray(self._data)

    def pmf(self):
        values = {}
        for v in self._data:
            if v not in values:
                values[v] = 1
            else:
                values[v] += 1
        return values


class Trace:
    def __init__(self, data=None, mode='auto'):
        if data is not None:
            try:
                valid_as_samples = all(len(item) == 2 for item in data)
                valid_as_split = len(data) == 2 and len(data[0]) == len(data[1])
            except TypeError as e:
                raise ValueError('wrong data shape') from e
            else:
                if not valid_as_samples and not valid_as_split:
                    raise ValueError('wrong data shape')

                if (mode == 'auto' and valid_as_samples) or mode == 'samples':
                    _data = [(t, v) for (t, v) in data]
                elif mode in {'auto', 'split'}:
                    _data = [(t, v) for t, v in zip(*data)]
                else:
                    raise ValueError('invalid mode')

                ar = np.asarray(_data).transpose()[0]
                if np.any((ar[1:] - ar[:-1]) < 0):
                    raise ValueError('data must be ordered by time')

                self._data = _data
        else:
            self._data = []

    def record(self, t, v):
        if self._data and t < self._data[-1][0]:
            raise ValueError('adding data in past prohibited')
        self._data.append((t, v),)

    @property
    def empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def pmf(self):
        if self.empty:
            raise ValueError('expected non-empty values')
        values = {}
        for i in range(0, len(self._data) - 1):
            v, dt = self._data[i][1], self._data[i + 1][0] - self._data[i][0]
            if self._data[i][1] not in values:
                values[v] = dt
            else:
                values[v] += dt
        total_time = sum(values.values())
        values = {v: t / total_time for v, t in values.items()}
        return values

    def timeavg(self):
        return sum(v * p for v, p in self.pmf().items())

    def _convert(self, fn, mode):
        if mode == 'samples':
            return fn(fn([t, v]) for (t, v) in self._data)
        elif mode == 'split':
            timestamps, values = [], []
            for (t, v) in self._data:
                timestamps.append(t)
                values.append(v)
            if timestamps:
                return fn([fn(timestamps), fn(values)])
            return fn()
        else:
            raise ValueError('invalid mode')

    def as_list(self, mode='samples'):
        return self._convert(list, mode)

    def as_tuple(self, mode='samples'):
        return self._convert(tuple, mode)

    def asarray(self, mode='samples'):
        return np.asarray(self.as_list(mode))


class Intervals:
    def __init__(self, timestamps=None):
        if timestamps:
            _timestamps = [0] + list(timestamps)
            try:
                _zipped = zip(_timestamps[:-1], _timestamps[1:])
                if any(x > y for x, y in _zipped):
                    raise ValueError('timestamps must be ascending')
            except TypeError as e:
                raise TypeError('only numeric values expected') from e
            self._timestamps = [0] + list(timestamps)
        else:
            self._timestamps = [0]

    @property
    def last(self):
        return self._timestamps[-1]

    @property
    def empty(self):
        return len(self._timestamps) == 1

    def __len__(self):
        return len(self._timestamps) - 1

    def record(self, timestamp):
        try:
            if timestamp < self.last:
                raise ValueError('prohibited timestamps from past')
        except TypeError as e:
            raise TypeError('only numeric values expected') from e
        self._timestamps.append(timestamp)

    def statistic(self):
        return Statistic(self.as_tuple())

    def as_tuple(self):
        ar = np.asarray(self._timestamps)
        return tuple(ar[1:] - ar[:-1])

    def as_list(self):
        return list(self.as_tuple())
