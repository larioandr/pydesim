from unittest.mock import patch, ANY, Mock

import pytest

from pydesim import simulate, Model


def test_simulate_signature():
    ret = simulate({}, init=None, fin=None, handlers={}, stime_limit=1)
    assert ret.stime == 0
    assert ret.data == {}


def test_simulate_executes_init_and_fin():
    """In this test we validate that `simulate()` calls init and fin methods.
    """
    data = []

    def init(sim):
        sim.data.append(1)

    def fin(sim):
        sim.data.append('A')

    ret = simulate(data, init=init, fin=fin, handlers={}, stime_limit=1)

    assert ret.stime == 0
    assert ret.data == [1, 'A']


def test_simulate_accepts_classes_with_create_method():
    class ModelData:
        default_value = 'correct value'

        def __init__(self, value='incorrect value'):
            self.value = value

        @classmethod
        def create(cls):
            return ModelData(cls.default_value)

    ret_for_default_value = simulate(ModelData)

    ModelData.default_value = 'some new value'
    ret_for_updated_value = simulate(ModelData)

    assert ret_for_default_value.data.value == 'correct value'
    assert ret_for_updated_value.data.value == 'some new value'


def test_simulate_accept_classes_without_create():
    class ModelData:
        default_value = 'default value'

        def __init__(self):
            self.value = ModelData.default_value

    ret_for_default_value = simulate(ModelData)

    ModelData.default_value = 'new value'
    ret_for_updated_value = simulate(ModelData)

    assert ret_for_default_value.data.value == 'default value'
    assert ret_for_updated_value.data.value == 'new value'


def test_scheduled_methods_are_called_in_chain():
    def write_some_data(sim, value='first'):
        sim.data.append(value)
        if value == 'first':
            sim.schedule(3, write_some_data, args=('second',))
        elif value == 'second':
            sim.schedule(10, write_some_data, kwargs={'value': 'third'})

    def init(sim):
        sim.schedule(1, write_some_data)

    ret = simulate([], init=init)

    assert ret.stime == 14
    assert ret.data == ['first', 'second', 'third']


def test_handlers_can_be_passed_and_accessed_via_sim_handlers_field():
    def f1(sim):
        sim.data.append(1)
        sim.schedule(0, sim.handlers.get('second'))

    def f2(sim):
        sim.data.append(2)
        sim.schedule(0, sim.handlers.third)

    def f3(sim):
        sim.data.append(3)

    def init(sim):
        sim.schedule(0, sim.handlers['first'])

    ret = simulate([], init=init, handlers=dict(first=f1, second=f2, third=f3))

    assert ret.data == [1, 2, 3]


def test_schedule_multiple_events():
    def handler(sim):
        sim.data.append(sim.stime)

    def init(sim):
        sim.schedule(1, handler)
        sim.schedule(2, handler)

    ret = simulate([], init=init)

    assert ret.data == [1, 2]


def test_schedule_orders_events_by_time():
    def f(sim):
        sim.data.append(f'{int(sim.stime)}F')
        sim.schedule(1.0, g)

    def g(sim):
        sim.data.append(f'{int(sim.stime)}G')

    def h(sim):
        sim.data.append(f'{int(sim.stime)}H')

    def init(sim):
        sim.schedule(1.0, f)
        sim.schedule(4.0, f)
        sim.schedule(3.0, h)

    ret = simulate([], init)

    assert ret.data == ['1F', '2G', '3H', '4F', '5G']


def test_schedule_accept_none_handler_by_changing_only_time():
    def init(sim):
        sim.schedule(5)

    ret = simulate([], init=init)

    assert ret.stime == 5


def test_schedule_negative_delays_not_allowed():
    def invalid_init(sim):
        sim.schedule(-1)

    def invalid_handler(sim):
        sim.schedule(-0.1)

    def valid_init(sim):
        sim.schedule(10, invalid_handler)

    with pytest.raises(ValueError) as excinfo1:
        simulate([], init=invalid_init)

    with pytest.raises(ValueError) as excinfo2:
        simulate([], init=valid_init)

    assert "negative delay" in str(excinfo1.value).lower()
    assert "negative delay" in str(excinfo2.value).lower()


def test_stime_is_readonly():
    def valid_handler(sim):
        sim.data.append('OK')

    def valid_init(sim):
        sim.schedule(1, sim.handlers.handler)

    with pytest.raises(AttributeError) as excinfo1:
        def invalid_init(sim):
            sim.stime = 10

        simulate([], init=invalid_init)

    with pytest.raises(AttributeError) as excinfo2:
        def invalid_handler(sim):
            sim.stime += 1

        simulate([], init=valid_init, handlers={'handler': invalid_handler})

    with pytest.raises(AttributeError) as excinfo3:
        def invalid_fin(sim):
            sim.stime -= 1

        simulate([], init=valid_init, fin=invalid_fin, handlers={
            'handler': valid_handler
        })

    assert 'set attribute' in str(excinfo1.value)
    assert 'set attribute' in str(excinfo2.value)
    assert 'set attribute' in str(excinfo3.value)


def test_sim_provide_cancel_operation():
    def init(sim):
        eid = sim.schedule(1)
        sim.cancel(eid)

    ret = simulate([], init)

    assert ret.stime == 0
    assert ret.num_events == 0


def test_simulate_with_stime_limit():
    def f(sim):
        sim.data.append('f')
        sim.schedule(2, g)

    def g(sim):
        sim.data.append('g')

    def init(sim):
        sim.schedule(1, f)
        sim.data.append('init')

    ret1 = simulate([], init, stime_limit=0.5)
    ret2 = simulate([], init, stime_limit=1)
    ret3 = simulate([], init, stime_limit=2)
    ret4 = simulate([], init, stime_limit=3)

    assert ret1.data == ['init']
    assert ret2.data == ['init', 'f']
    assert ret3.data == ['init', 'f']
    assert ret4.data == ['init', 'f', 'g']

    assert ret1.num_events == 0
    assert ret2.num_events == 1
    assert ret3.num_events == 1
    assert ret4.num_events == 2

    assert ret1.stime == 1
    assert ret2.stime == 3
    assert ret3.stime == 3
    assert ret4.stime == 3


def test_simulate_accepts_params():
    params = {'x': 10, 'y': 'hello'}
    with patch('pydesim.simulator.Simulator') as SimulatorMock:
        simulate([], params=params)
        SimulatorMock.assert_called_with(ANY, [], ANY, params, ANY)


def test_params_accessible_via_getattr_and_getitem():
    params = {'x': 10, 'y': 'hello'}

    def init(sim):
        assert sim.params.x == 10
        assert sim.params['x'] == 10
        assert sim.params.y == 'hello'

    simulate([], init=init, params=params)


def test_array_returned_when_params_are_given_in_array():
    params = [{'x': 1}, {'x': 2}]
    data_class = Mock()
    result = simulate(data_class, params=params)
    assert result[0].params.as_dict() == {'x': 1}
    assert result[1].params.as_dict() == {'x': 2}


def test_simulate_calls_constructor_without_parameters_but_with_sim():
    with patch('pydesim.simulator.Simulator') as SimulatorMock:
        class SomeModel(Model):
            def __init__(self, sim):
                assert isinstance(sim, SimulatorMock)
                super().__init__(self, sim)
                assert sim.params.x == 10
                assert sim.params.y == 'hello'
        
        result = simulate(SomeModel, params={'x': 10, 'y': 'hello'})
