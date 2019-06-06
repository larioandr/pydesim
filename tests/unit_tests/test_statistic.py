import numpy as np
import pytest

from pydesim import Statistic


def test_statistic_is_initially_empty():
    st = Statistic()
    assert len(st) == 0
    assert st.empty
    assert st.as_list() == []
    assert st.as_tuple() == ()


def test_statistic_can_be_created_from_list():
    data = [1, 2, 3]
    st = Statistic(data)
    assert len(st) == 3
    assert not st.empty
    assert st.as_list() == [1, 2, 3]
    assert st.as_tuple() == tuple(data)
    assert st.as_list() is not data  # check that data was copied


def test_statistic_copy_data_content_instead_of_pointer():
    data = [1, 2]
    st = Statistic(data)
    st.append(3)
    assert st.as_tuple() == (1, 2, 3)
    assert data == [1, 2]


def test_statistic_array_converter():
    st = Statistic([1, 2, 3])
    np.testing.assert_equal(st.asarray(), np.asarray([1, 2, 3]))
    assert isinstance(st.asarray(), np.ndarray)


def test_statistic_append_adds_values():
    st = Statistic()
    st.append(1)
    st.append(20)
    assert st.as_tuple() == (1, 20)


def test_statistic_extend_adds_all_items():
    st = Statistic([1])
    st.extend([2, 3])
    assert st.as_tuple() == (1, 2, 3)


def test_statistic_extend_raises_error_when_noniterable_passed():
    st = Statistic()
    with pytest.raises(TypeError) as excinfo:
        st.extend(1)
    assert 'not iterable' in str(excinfo.value).lower()


def test_statistic_mean():
    st = Statistic()
    with pytest.raises(ValueError) as excinfo:
        st.mean()
    assert 'no data' in str(excinfo.value).lower()

    st = Statistic([1])
    assert st.mean() == 1

    st.extend([2, 3])
    assert st.mean() == 2


def test_statistic_std():
    st = Statistic()
    with pytest.raises(ValueError) as excinfo:
        st.std()
    assert 'no data' in str(excinfo.value).lower()

    st = Statistic([1])
    assert st.std() == 0

    st.extend([2, 3])
    np.testing.assert_allclose(st.std(), (2/3)**0.5, atol=0.001)


def test_statistic_var():
    st = Statistic()
    with pytest.raises(ValueError) as excinfo:
        st.var()
    assert 'no data' in str(excinfo.value).lower()

    st = Statistic([1])
    assert st.var() == 0

    st.extend([2, 3])
    np.testing.assert_allclose(st.var(), 2/3, atol=0.001)


def test_statistic_moment_raises_error_when_called_for_empty_statistic():
    st = Statistic()
    with pytest.raises(ValueError) as excinfo:
        st.moment(1)
    assert 'no data' in str(excinfo.value).lower()


def test_statistic_moment_raises_error_when_passed_zero_negative_or_float():
    st = Statistic([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        st.moment(0)
    assert 'positive integer expected' in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo:
        st.moment(-1)
    assert 'positive integer expected' in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo:
        st.moment(1.5)
    assert 'positive integer expected' in str(excinfo.value).lower()


def test_statistic_moment_evaluation():
    st = Statistic([1])
    assert st.moment(1) == 1
    assert st.moment(2) == 1

    st.extend([2, 3, 4])
    assert st.moment(1) == 2.5
    assert st.moment(2) == 7.5


def test_lag_k_raises_error_when_called_for_empty_statistic():
    st = Statistic()
    with pytest.raises(ValueError) as excinfo:
        st.lag(1)
    assert 'no data' in str(excinfo.value).lower()


def test_lag_k_raises_error_when_passed_negative_or_float():
    st = Statistic([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        st.lag(-1)
    assert 'non-negative integer expected' in str(excinfo.value).lower()
    with pytest.raises(ValueError) as excinfo:
        st.lag(2.5)
    assert 'non-negative integer expected' in str(excinfo.value).lower()


def test_lag_k_raises_error_when_k_greater_then_length():
    st = Statistic([1, 2])
    error_message = 'statistic has too few samples'
    with pytest.raises(ValueError) as excinfo1:
        st.lag(2)
    with pytest.raises(ValueError) as excinfo2:
        st.lag(3)
    assert error_message in str(excinfo1.value).lower()
    assert error_message in str(excinfo2.value).lower()


def test_lag_0_is_always_1():
    st1 = Statistic([10])
    st2 = Statistic([-1, 1] * 5)
    st3 = Statistic(np.random.exponential(1, 7))
    assert st1.lag(0) == 1
    assert st2.lag(0) == 1
    assert st3.lag(0) == 1


def test_lag_k_estimation_for_positive_k():
    data = [-1, 0, 1, 0] * 10
    st = Statistic(data)
    np.testing.assert_almost_equal(st.lag(0), 1)
    np.testing.assert_almost_equal(st.lag(1), 0)
    np.testing.assert_almost_equal(st.lag(2), -1)
    np.testing.assert_almost_equal(st.lag(3), 0)
