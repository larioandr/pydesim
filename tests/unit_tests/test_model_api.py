from unittest.mock import Mock

import pytest

from pydesim import Model


class Ping(Model):
    """Ping dummy model will be used when we will need a dummy model.
    """
    def __init__(self, sim):
        super().__init__(sim, parent=None)


#############################################################################
# TEST MODULE CONNECTIONS
#############################################################################
def test_creating_single_connections():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.connections['pong'] = pong_mock

    assert ping.connections['pong'].module == pong_mock


def test_creating_multiple_connections_with_update():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock = Mock(), Mock()
    ping.connections.update({'red': red_mock, 'blue': blue_mock})

    assert ping.connections['red'].module == red_mock
    assert ping.connections['blue'].module == blue_mock


def test_listing_connections():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock = Mock(), Mock()
    ping.connections.update({'red': red_mock, 'blue': blue_mock})

    assert set(ping.connections.names()) == {'red', 'blue'}
    assert set(ping.connections.modules()) == {red_mock, blue_mock}
    assert dict(ping.connections.as_dict()) == {
        'red': red_mock, 'blue': blue_mock
    }


def test_getting_connection_with_get_method():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.connections['pong'] = pong_mock
    assert ping.connections.get('pong', None).module == pong_mock
    assert ping.connections.get('wrong_name', 'something') == 'something'
    assert ping.connections.get('wrong_name') is None


def test_connection_exists():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.connections['pong'] = pong_mock

    assert 'pong' in ping.connections
    assert 'wrong_name' not in ping.connections


def test_getting_wrong_connection_with_getitem_raises_error():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    with pytest.raises(KeyError):
        print(ping.connections['pong'])


def test_connection_method_send():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    message_mock = Mock()
    ping.connections['pong'] = pong_mock
    ping.connections['pong'].send(message_mock)

    sim_mock.schedule.assert_called_with(
        0, pong_mock.handle_message, args=(message_mock,),
        kwargs={'sender': ping}
    )


def test_connection_delay_setting():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock = Mock(), Mock()
    message_mock = Mock()
    ping.connections.update({'red': red_mock, 'blue': blue_mock})
    ping.connections['red'].delay = 13
    ping.connections['blue'].delay = lambda: 42

    ping.connections['red'].send(message_mock)
    sim_mock.schedule.assert_called_with(
        13, red_mock.handle_message, args=(message_mock,),
        kwargs={'sender': ping}
    )

    ping.connections['blue'].send(message_mock)
    sim_mock.schedule.assert_called_with(
        42, blue_mock.handle_message, args=(message_mock,),
        kwargs={'sender': ping}
    )


#############################################################################
# TEST CHILDREN MANAGER API
#############################################################################
def test_adding_single_child():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.children['pong'] = pong_mock

    assert ping.children['pong'] == pong_mock
    # noinspection PyProtectedMember
    pong_mock._set_parent.assert_called_with(ping)


# noinspection PyProtectedMember
def test_replacing_single_child():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock = Mock(), Mock()
    ping.children['pong'] = red_mock
    assert ping.children['pong'] == red_mock
    red_mock._set_parent.assert_called_with(ping)

    ping.children['pong'] = blue_mock
    assert ping.children['pong'] == blue_mock
    red_mock._set_parent.assert_called_with(None)
    blue_mock._set_parent.assert_called_with(ping)


def test_adding_children_array():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pongs = [Mock(), Mock(), Mock()]
    ping.children['pong'] = pongs

    assert ping.children['pong'] == tuple(pongs)
    for pong in pongs:
        # noinspection PyProtectedMember
        pong._set_parent.assert_called_with(ping)


# noinspection PyProtectedMember
def test_filling_children_with_update():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock, green_mock, pink_mock = Mock(), Mock(), Mock(), Mock()
    all_mocks = (red_mock, blue_mock, green_mock, pink_mock)
    ping.children.update({
        'red': red_mock,
        'blue': blue_mock,
        'colors': (green_mock, pink_mock),
    })

    assert ping.children['red'] == red_mock
    assert ping.children['blue'] == blue_mock
    assert ping.children['colors'] == (green_mock, pink_mock)

    for mock in all_mocks:
        mock._set_parent.assert_called_with(ping)


def test_getting_children_with_get_method():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.children['pong'] = pong_mock

    assert ping.children.get('pong') == pong_mock
    assert ping.children.get('xxx', 42) == 42
    assert ping.children.get('xxx') is None


def test_checking_children_exists():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    pong_mock = Mock()
    ping.children['pong'] = pong_mock

    assert 'pong' in ping.children
    assert 'xxx' not in ping.children


def test_getting_all_children():
    sim_mock = Mock()
    ping = Ping(sim_mock)

    red_mock, blue_mock, green_mock, pink_mock = Mock(), Mock(), Mock(), Mock()
    ping.children.update({
        'red': red_mock,
        'blue': blue_mock,
        'others': [green_mock, pink_mock],
    })

    assert set(ping.children.all()) == {
        red_mock, blue_mock, green_mock, pink_mock}
