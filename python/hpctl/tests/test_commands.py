from collections import OrderedDict
import pytest
from mock import MagicMock
from hpctl.core import process_command

@pytest.fixture
def mocks():
    return OrderedDict([
        ('be', MagicMock()),
        ('fe', MagicMock()),
        ('sch', MagicMock()),
        ('res', MagicMock()),
        ('xpctl', MagicMock()),
    ])


def test_command_is_none(mocks):
    process_command(None, *mocks.values())
    mocks['be'].kill.assert_not_called()
    mocks['res'].set_killed.assert_not_called()
    mocks['sch'].remove.assert_not_called()
    mocks['fe'].update.assert_not_called()


def test_command_kill(mocks):
    label = '12345'
    command = {'command': 'kill', 'label': label}
    process_command(command, *mocks.values())
    mocks['be'].kill.assert_called_once_with(label)
    mocks['res'].set_killed.assert_called_once_with(label)
    mocks['sch'].remove.assert_called_once_with(label)
    mocks['fe'].update.assert_called_once()


def text_command_kill_no_scheduler(mocks):
    label = '12345'
    command = {'command': 'kill', 'label': label}
    process_command(command, mocks['be'], mocks['fe'], None, mocks['res'], mocks['xpctl'])
    mocks['be'].kill.assert_called_once_with(label)
    mocks['res'].set_killed.assert_called_once_with(label)
    mocks['fe'].update.assert_called_once()


def test_command_launch(mocks):
    label = '12345'
    exp = 'exp'
    config = 'config'
    command = {
        'command': 'launch',
        'label': label,
        'experiment_config': exp,
        'config': config
    }
    process_command(command, *mocks.values())
    mocks['res'].add_experiment.assert_called_once_with(exp)
    mocks['sch'].add.assert_called_once_with(label, command)
    mocks['res'].insert.assert_called_once_with(label, config)
    mocks['res'].save.assert_called_once()


def test_command_launch_no_exp(mocks):
    label = '12345'
    config = 'config'
    command = {
        'command': 'launch',
        'label': label,
        'config': config
    }
    process_command(command, *mocks.values())
    mocks['res'].add_experiment.assert_not_called()
    mocks['sch'].add.assert_called_once_with(label, command)
    mocks['res'].insert.assert_called_once_with(label, config)
    mocks['res'].save.assert_called_once()


def test_command_xpctl(mocks):
    id_ = 'id'
    label = '12345'
    command = {'label': label, 'command': 'xpctl'}
    xpctl = MagicMock()
    xpctl.put_result.return_value = id_
    results = MagicMock()
    results.get_xpctl.return_value = False
    process_command(command, mocks['be'], mocks['fe'], None, results, xpctl)
    results.get_xpctl.assert_called_once_with(label)
    xpctl.put_result.assert_called_once_with(label)
    results.set_xpctl.assert_called_once_with(label, id_)

def test_command_xpctl_already(mocks):
    id_ = 'id'
    label = '12345'
    command = {'label': label, 'command': 'xpctl'}
    xpctl = MagicMock()
    xpctl.put_result.return_value = id_
    results = MagicMock()
    results.get_xpctl.return_value = True
    process_command(command, mocks['be'], mocks['fe'], None, results, xpctl)
    results.get_xpctl.assert_called_once_with(label)
    xpctl.put_result.assert_not_called()
    results.set_xpctl.assert_not_called()
