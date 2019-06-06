import numpy as np
import pytest

from pydesim import Trace


#
# Validate Trace object instantiation with no data, with valid data and with
# invalid data.
# Also check that expected exceptions are raised on some typical illegal data.
#
def test_trace_is_initially_empty():
    trace = Trace()
    assert trace.empty
    assert len(trace) == 0
    assert trace.as_tuple() == ()


def test_trace_creation_with_valid_values_and_timestamps():
    trace = Trace([[1, 2, 3], [10, 11, 9]])
    assert not trace.empty
    assert len(trace) == 3
    assert trace.as_tuple() == ((1, 10), (2, 11), (3, 9))


def test_trace_creation_with_valid_data_given_in_pairs():
    data = [(1, 5), (2, 8), (3, 40), (9, 34)]
    trace = Trace(data)
    assert not trace.empty
    assert len(trace) == 4
    assert trace.as_tuple() == tuple(data)


@pytest.mark.parametrize('data', [[(1, 10)], [(1,), (10,)]])
def test_trace_creation_with_single_item(data):
    trace = Trace(data)
    assert not trace.empty
    assert len(trace) == 1
    assert trace.as_tuple() == ((1, 10),)


@pytest.mark.parametrize('data', [
    [1],  # less dimensions then needed
    [1, 2],  # also less dimensions then needed
    [(1, 2, 3)],  # more dimensions then needed
    [(1, 2), (3,)],  # some item has less dimensions then needed
    [(1, 2), (3, 4, 5)],  # some item has more dimensions then needed
])
def test_trace_creation_raises_error_when_passed_data_with_wrong_shape(data):
    with pytest.raises(ValueError) as excinfo:
        Trace(data)
    assert 'wrong data shape' in str(excinfo.value).lower()


@pytest.mark.parametrize('mode,expected', [
    ('split', ((1, 10), (2, 20))),
    ('samples', ((1, 2), (10, 20))),
])
def test_trace_creation_with_2x2_data_using_mode_argument(mode, expected):
    trace = Trace(((1, 2), (10, 20)), mode=mode)
    assert trace.as_tuple() == expected


def test_trace_creation_with_2x2_data_without_mode_is_the_same_as_samples():
    data = ((1, 2), (10, 20))  # t=1, v=10; t=2, v=20
    trace_default = Trace(data)
    trace_samples = Trace(data, mode='samples')
    assert trace_default.as_tuple() == trace_samples.as_tuple()


def test_trace_creation_with_wrong_mode_raises_error():
    with pytest.raises(ValueError) as excinfo:
        Trace([(1,), (2,)], mode='wrong')
    assert 'invalid mode' in str(excinfo.value).lower()


@pytest.mark.parametrize('data', [
    [(2, 13), (3, 5), (1, 4)],
    [(1, 3, 2), (4, 5, 13)],
])
def test_trace_creation_with_unordered_timestamps_raises_error(data):
    with pytest.raises(ValueError) as excinfo:
        Trace(data)
    assert 'data must be ordered by time' in str(excinfo.value).lower()


#
# Test data recording
#
def test_record_adds_data():
    trace = Trace()
    assert trace.as_tuple() == ()

    trace.record(t=1, v=10)
    assert trace.as_tuple() == ((1, 10),)

    trace.record(t=2, v=13)
    assert trace.as_tuple() == ((1, 10), (2, 13), )


def test_record_with_past_time_causes_error():
    trace = Trace([(10, 13)])
    with pytest.raises(ValueError) as excinfo:
        trace.record(5, 10)
    assert 'adding data in past prohibited' in str(excinfo.value).lower()


#
# Test estimation of PMF and time average:
#
def test_getting_pmf_for_empty_trace_raises_error():
    trace = Trace()
    with pytest.raises(ValueError) as excinfo:
        trace.pmf()
    assert 'expected non-empty values' in str(excinfo.value).lower()


@pytest.mark.parametrize('data, probs', [
    ([(0, 0), (1, 10), (4, 20), (5, 0)], {0: 0.2, 10: 0.6, 20: 0.2}),
    ([(0, 0), (1, 10), (2, 10), (4, 20), (5, 0)], {0: 0.2, 10: 0.6, 20: 0.2}),
    ([(0, 2), (8, 3), (11, 4), (12, 3), (16, 2)], {2: .5, 3: .4375, 4: .0625}),
])
def test_getting_pmf(data, probs):
    trace = Trace(data)
    pmf = trace.pmf()
    assert set(pmf.keys()) == set(probs.keys())

    estimated, expected = [], []
    for v, p in probs.items():
        estimated.append(pmf[v])
        expected.append(p)
    np.testing.assert_almost_equal(estimated, expected)


@pytest.mark.parametrize('data', [
    [(0, 0), (1, 10), (4, 20), (5, 0)],
    [(0, 0), (1, 10), (2, 10), (4, 20), (5, 0)],
    [(0, 2), (8, 3), (11, 4), (12, 3), (16, 2)],
])
def test_getting_timeavg(data):
    def estimate():
        interval = data[-1][0] - data[0][0]
        acc = 0
        for i in range(1, len(data)):
            acc += data[i-1][1] * (data[i][0] - data[i-1][0]) / interval
        return acc

    trace = Trace(data)
    estimated_avg = estimate()
    np.testing.assert_almost_equal(trace.timeavg(), estimated_avg)


#
# Test getters and converters
#
@pytest.mark.parametrize('data, mode, expected', [
    (None, 'samples', []),
    (None, 'split', []),
    ([[0, 0], [1, 5], [2, 7]], 'samples', None),
    ([[0, 0], [1, 5], [2, 7]], 'split', [[0, 1, 2], [0, 5, 7]]),
    ([[1, 13], [8, 42], [12, 34], [13, 51]], 'samples', None),
    ([[1, 13], [8, 42], [12, 34], [13, 51]], 'split',
     [[1, 8, 12, 13], [13, 42, 34, 51]]),
])
def test_as_list(data, mode, expected):
    trace = Trace(data)
    if expected is None:
        expected = data
    assert trace.as_list(mode=mode) == expected


def test_as_list_with_illegal_mode_raises_error():
    trace = Trace([[0, 0], [1, 5]])
    with pytest.raises(ValueError) as excinfo:
        trace.as_list('wrong mode')
    assert 'invalid mode' in str(excinfo.value).lower()


@pytest.mark.parametrize('data, mode, expected', [
    (None, 'samples', ()),
    (None, 'split', ()),
    (((0, 0), (1, 5), (2, 7)), 'samples', None),
    (((0, 0), (1, 5), (2, 7)), 'split', ((0, 1, 2), (0, 5, 7))),
    (((1, 13), (8, 42), (12, 34), (13, 51)), 'samples', None),
    (((1, 13), (8, 42), (12, 34), (13, 51)), 'split',
     ((1, 8, 12, 13), (13, 42, 34, 51))),
])
def test_as_tuple(data, mode, expected):
    trace = Trace(data)
    if expected is None:
        expected = data
    assert trace.as_tuple(mode=mode) == expected


def test_as_tuple_with_illegal_mode_raises_error():
    trace = Trace([[0, 0], [1, 5]])
    with pytest.raises(ValueError) as excinfo:
        trace.as_tuple('wrong mode')
    assert 'invalid mode' in str(excinfo.value).lower()


@pytest.mark.parametrize('data, mode, expected', [
    (None, 'samples', np.asarray([])),
    (None, 'split', np.asarray([])),
    (((0, 0), (1, 5), (2, 7)), 'samples', np.asarray([(0, 0), (1, 5), (2, 7)])),
    (((0, 0), (1, 5), (2, 7)), 'split', np.asarray([(0, 1, 2), (0, 5, 7)])),
    (((1, 13), (8, 42), (12, 34), (13, 51)), 'samples',
     np.asarray([(1, 13), (8, 42), (12, 34), (13, 51)])),
    (((1, 13), (8, 42), (12, 34), (13, 51)), 'split',
     np.asarray([(1, 8, 12, 13), (13, 42, 34, 51)])),
])
def test_asarray(data, mode, expected):
    trace = Trace(data)
    actual = trace.asarray(mode)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_almost_equal(actual, expected)


def test_asarray_with_illegal_mode_raises_error():
    trace = Trace([[0, 0], [1, 5]])
    with pytest.raises(ValueError) as excinfo:
        trace.asarray('wrong mode')
    assert 'invalid mode' in str(excinfo.value).lower()
