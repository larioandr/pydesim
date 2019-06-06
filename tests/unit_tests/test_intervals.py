import pytest
from numpy import asarray
from numpy.testing import assert_almost_equal

from pydesim import Intervals


#
# Test Intervals instantiation:
#
def test_empty_intervals_instantiation():
    ints = Intervals()
    assert ints.empty
    assert len(ints) == 0
    assert ints.as_tuple() == ()
    assert ints.last == 0


@pytest.mark.parametrize('timestamps', [[1], [2.3, 4.5], (0, 0, 1)])
def test_intervals_creation_with_valid_data(timestamps):
    ints = Intervals(timestamps)
    _timestamps = asarray([0] + list(timestamps))
    assert not ints.empty
    assert len(ints) == len(timestamps)
    assert ints.as_tuple() == tuple(_timestamps[1:] - _timestamps[:-1])
    assert ints.last == timestamps[-1]


def test_intervals_copy_list_content_instead_of_pointer():
    data = [1, 2]
    ints = Intervals(data)
    ints.record(3)
    assert ints.as_tuple() == (1, 1, 1)
    assert data == [1, 2]


def test_intervals_accept_only_timestamps_in_future():
    with pytest.raises(ValueError) as excinfo:
        Intervals([1, 2, 1.9, 3])
    assert 'timestamps must be ascending' in str(excinfo.value).lower()


@pytest.mark.parametrize('timestamps', [[1.2, 'hello'], [(1,)], [complex(1)]])
def test_data_of_illegal_types_causes_type_error(timestamps):
    with pytest.raises(TypeError) as excinfo:
        Intervals(timestamps)
    assert 'only numeric values expected' in str(excinfo.value).lower()


#
# Test Intervals data modification API
#
def test_record_valid_data():
    ints = Intervals()
    assert ints.as_tuple() == ()

    ints.record(2)
    assert_almost_equal(ints.as_tuple(), (2,))

    ints.record(3.4)
    assert_almost_equal(ints.as_tuple(), (2, 1.4))


def test_record_in_past_raises_error():
    ints = Intervals([10])
    with pytest.raises(ValueError) as excinfo:
        ints.record(9.9)
    assert 'prohibited timestamps from past' in str(excinfo.value).lower()


@pytest.mark.parametrize('value', ['hello', (1,), complex(1)])
def test_record_illegal_types_raises_error(value):
    ints = Intervals()
    with pytest.raises(TypeError) as excinfo:
        ints.record(value)
    assert 'only numeric values expected' in str(excinfo.value).lower()


#
# Test converters
#
@pytest.mark.parametrize('timestamps', [(), (1,), (2, 3)])
def test_as_tuple(timestamps):
    ints = Intervals(timestamps)
    _timestamps = asarray([0] + list(timestamps))
    assert ints.as_tuple() == tuple(_timestamps[1:] - _timestamps[:-1])


@pytest.mark.parametrize('timestamps', [(), (1,), (2, 3)])
def test_as_list(timestamps):
    ints = Intervals(timestamps)
    _timestamps = asarray([0] + list(timestamps))
    assert ints.as_list() == list(_timestamps[1:] - _timestamps[:-1])


@pytest.mark.parametrize('data', [(), (1,), (2, 3)])
def test_statistic(data):
    ints = Intervals(data)
    stats = ints.statistic()
    assert_almost_equal(stats.as_tuple(), ints.as_tuple())
