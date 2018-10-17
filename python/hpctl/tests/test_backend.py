import pytest
import numpy as np
from mock import MagicMock
from hpctl.backend import LocalGPUBackend
from hpctl.results import States


def test_request_gpus_none_open():
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': 'taken', '1': 'taken'}
    gpus = be._request_gpus(1)
    assert gpus is None


def test_request_gpus_one():
    gold = ['1']
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': 'taken', '1': None}
    gpus = be._request_gpus(1)
    assert gpus == gold


def test_request_gpus_one_multi_open():
    gold = [['1'], ['0']]
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None}
    gpus = be._request_gpus(1)
    assert gpus in gold


def test_request_gpus_multi():
    gold = ['0', '1']
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None}
    gpus = be._request_gpus(2)
    assert gpus == gold


def test_request_gpus_multi_extra_open():
    gold = {'0', '1', '3'}
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None, '2': 'taken', '3': None}
    gpus = be._request_gpus(2)
    for gpu in gpus:
        assert gpu in gold


def test_request_gpus_not_enough():
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': 'taken', '2': 'taken'}
    gpus = be._request_gpus(2)
    assert gpus is None


def test_reserve_gpus():
    be = LocalGPUBackend()
    num_gpus = np.random.randint(5, 10)
    be.gpus_to_job = {i: None for i in range(num_gpus)}
    gpus = list(map(str, np.random.choice(np.arange(num_gpus), replace=False, size=np.random.randint(1, num_gpus -1 ))))
    job = 'job'
    be._reserve_gpus(gpus, job)
    for gpu in gpus:
        assert be.gpus_to_job[gpu] == job
    for gpu, value in be.gpus_to_job.items():
        if gpu not in gpus:
            assert value is None


def test_kill():
    gold = MagicMock()
    dummy = MagicMock()
    be = LocalGPUBackend()
    be._free_resources = MagicMock()
    be.label_to_job = {'gold': gold, 'dummy': dummy}
    be.kill('gold')
    gold.stop.assert_called_once()
    gold.join.assert_called_once()
    dummy.stop.assert_not_called()
    dummy.join.assert_not_called()
    be._free_resources.assert_called_once()


def test_kill_not_there():
    gold = MagicMock()
    dummy = MagicMock()
    be = LocalGPUBackend()
    be._free_resources = MagicMock()
    be.label_to_job = {'gold': gold, 'dummy': dummy}
    be.kill('missing')
    gold.stop.assert_not_called()
    gold.join.assert_not_called()
    dummy.stop.assert_not_called()
    dummy.join.assert_not_called()
    be._free_resources.assert_not_called()


def RunningJob():
    job = MagicMock()
    job.is_done = False
    return job


def DoneJob():
    job = MagicMock()
    job.is_done = True
    return job


def FailedJob():
    job = MagicMock()
    job.is_done = True
    return job


def test_any_done_none():
    be = LocalGPUBackend()
    be.gpus_to_job = {'a': 'taken', 'b': 'taken'}
    be.jobs = [RunningJob(), RunningJob()]
    assert be.any_done() == False

def test_any_done_open_gpu():
    be = LocalGPUBackend()
    be.gpus_to_job = {'a': 'taken', 'b': None}
    be.jobs = [RunningJob()]
    assert be.any_done() == True

def test_any_done_open_job():
    be = LocalGPUBackend()
    be.gpus_to_job = {'a': 'taken', 'b': 'taken'}
    be.jobs = [RunningJob(), DoneJob()]
    assert be.any_done() == True

def test_any_done_no_jobs():
    be = LocalGPUBackend()
    be.gpus_to_job = {'a': 'taken'}
    be.jobs = []
    assert be.any_done() == True


def test_free_resources_none_done():
    be = LocalGPUBackend()
    one_job = RunningJob()
    one_label = 'a'
    one_gpu = '0'
    two_job = RunningJob()
    two_label = 'b'
    two_gpu = '1'
    be.jobs = [one_job, two_job]
    be.gpus_to_job = {one_gpu: one_job, two_gpu: two_job}
    be.label_to_job = {one_label: one_job, two_label: two_job}
    be._free_resources()
    assert one_job in be.jobs
    assert two_job in be.jobs
    assert be.gpus_to_job[one_gpu] == one_job
    assert be.gpus_to_job[two_gpu] == two_job
    assert one_label in be.label_to_job
    assert two_label in be.label_to_job
    for job in be.jobs:
        job.join.assert_not_called()


def test_free_resources_one_done():
    be = LocalGPUBackend()
    done_job = DoneJob()
    done_label = 'a'
    done_gpu = '0'
    run_job = RunningJob()
    run_label = 'b'
    run_gpu = '1'
    be.jobs = [done_job, run_job]
    be.gpus_to_job = {done_gpu: done_job, run_gpu: run_job}
    be.label_to_job = {done_label: done_job, run_label: run_job}
    be._free_resources()
    assert done_job not in be.jobs
    assert run_job in be.jobs
    assert be.gpus_to_job[done_gpu] == None
    assert be.gpus_to_job[run_gpu] == run_job
    assert run_label in be.label_to_job
    done_job.join.assert_called_once()
    run_job.join.assert_not_called()


def test_free_resources_multi_done():
    be = LocalGPUBackend()
    done_jobs = [DoneJob(), DoneJob()]
    done_labels = ['a', 'c']
    done_gpus = ['0', '2']
    run_job = RunningJob()
    run_label = 'b'
    run_gpu = '1'
    be.jobs = done_jobs + [run_job]
    be.gpus_to_job = {gpu: job for gpu, job in zip(done_gpus, done_jobs)}
    be.gpus_to_job[run_gpu] = run_job
    be.label_to_job = {label: job for label, job in zip(done_labels, done_jobs)}
    be.label_to_job[run_label] = run_job
    be._free_resources()
    for done_job in done_jobs:
        assert done_job not in be.jobs
    assert run_job in be.jobs
    for done_gpu in done_gpus:
        assert be.gpus_to_job[done_gpu] == None
    assert be.gpus_to_job[run_gpu] == run_job
    assert run_label in be.label_to_job
    for done_label in done_labels:
        done_job.join.assert_called_once()
    run_job.join.assert_not_called()


def test_delete_backend():
    be = LocalGPUBackend()
    jobs = [np.random.choice([RunningJob, DoneJob])() for _ in range(np.random.randint(2, 6))]
    be.jobs = jobs
    del be
    for job in jobs:
        job.join.assert_called_once()


def test_all_done_no_jobs():
    be = LocalGPUBackend()
    assert be.all_done(MagicMock()) == True


def test_all_done_all_done():
    be = LocalGPUBackend()
    be.labels = ['0', '1']
    result = MagicMock()
    result.get_state.return_value = States.DONE
    assert be.all_done(result) == True


def test_all_some_not():
    be = LocalGPUBackend()
    be.labels = ['0', '1']
    assert be.all_done(MagicMock()) == False


def test_all_done_remove_done():
    be = LocalGPUBackend()
    be.labels = ['0', '1']
    be.label_to_job = {x: None for x in be.labels}
    res = MagicMock()
    res.get_state = lambda x: States.DONE if x == '0' else None
    be.all_done(res)
    assert '0' not in be.label_to_job
    assert '1' in be.label_to_job


def test_all_done_remove_failed():
    be = LocalGPUBackend()
    be.labels = ['0', '1']
    be.label_to_job = {'0': FailedJob(), '1': None}
    res = MagicMock()
    res.get_state = lambda x: States.KILLED if x == '0' else None
    be.all_done(res)
    res.set_killed.assert_called_once_with('0')
    assert '0' not in be.label_to_job
    assert '1' in be.label_to_job


def test_launch():
    be = LocalGPUBackend()
    label = 'a'
    be.launch(label, None)
    assert be.labels == [label]
