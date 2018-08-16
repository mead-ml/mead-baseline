import os
import re
import sys
import json
import shutil
import sqlite3
import tempfile
import datetime
from pprint import pprint
from copy import deepcopy
from collections import namedtuple
from subprocess import check_output
from multiprocessing import Process, cpu_count, Manager
import numpy as np
import mead
import baseline
from baseline.progress import create_progress_bar
from baseline.utils import read_config_file, suppress_output
from xpctl.helpers import order_json

Version = namedtuple('Version', 'major minor patch')
simple_version_regex = re.compile('^(\d+)\.(\d+)\.(\d+)')

FRAMEWORKS = ['tensorflow', 'pytorch', 'dynet', 'keras']
FULL_FRAMEWORKS = ['tensorflow', 'pytorch', 'dynet']
TASKS = ['classify', 'tagger', 'seq2seq', 'lm']
PHASES = ['Train', 'Valid', 'Test']


def run_model(si, config_params, logs, settings, datasets, embeddings, task_name, dir_, gpu):
    """Run a model and collect system information."""
    os.chdir(dir_)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config_params['reporting'] = {}
    with suppress_output():
        task = mead.Task.get_task_specific(task_name, logs, settings, time=True)
        task.read_config(config_params, datasets)
        task.initialize(embeddings)
        task.train()

        si['framework_version'] = get_framework_version(config_params['backend'])
        si['cuda'], si['cudnn'] = get_cuda_version()
        # Anaconda pytorch comes with its own cudnn so I have to get it this way
        if config_params['backend'] == 'pytorch':
            import torch
            # si['cudnn'] = torch.backends.cudnn.version()
            si['cudnn'] = Version(7, 1, 2)
        si['gpu_name'], si['gpu_mem'] = get_gpu_info(gpu)
        si['cpu_name'], si['cpu_mem'], si['cpu_cores'] = get_cpu_info()
        si['python'] = get_python_version()
        si['baseline'] = version_str_to_tuple(baseline.__version__)


def parse_logs(file_name):
    """Read the timing logs out of the log file."""
    data = []
    with open(file_name) as f:
        for line in f:
            entry = json.loads(line)
            if 'time' in entry:
                data.append(entry)
    return data


def create_db(name="speed.db"):
    """Generate/Connect to the database to save results."""
    conn = sqlite3.connect(name, isolation_level='EXCLUSIVE')
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE speed(
                time FLOAT,
                phase TEXT,
                framework TEXT,
                framework_major INTEGER,
                framework_minor INTEGER,
                framework_patch INTEGER,
                dataset TEXT,
                model TEXT,
                task TEXT,
                config TEXT,
                cuda_major INTEGER,
                cuda_minor INTEGER,
                cuda_patch INTEGER,
                cudnn_major INTEGER,
                cudnn_minor INTEGER,
                cudnn_patch INTEGER,
                gpu_name TEXT,
                gpu_mem FLOAT,
                cpu_name TEXT,
                cpu_mem FLOAT,
                cpu_cores INTEGER,
                python_major INTEGER,
                python_minor INTEGER,
                python_patch INTEGER,
                baseline_major INTEGER,
                baseline_minor INTEGER,
                baseline_patch INTEGER,
                timestamp DATE
        );''')
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
    finally:
        c.close()
    return conn


def save_data(conn, speeds, config, si):
    """Write data to the db."""
    framework = config['backend']
    task = config['task']
    config_str = json.dumps(order_json(config)).encode('utf-8')
    task, dataset, model = get_run_info(config)
    fw_v = si['framework_version']
    bl_v = si['baseline']
    c_v = si['cuda']
    nn_v = si['cudnn']
    p_v = si['python']

    try:
        c = conn.cursor()
        for speed in speeds:
            c.execute('''
                INSERT INTO speed (
                    time, phase, framework, dataset, model, task, config,
                    framework_major, framework_minor, framework_patch,
                    baseline_major, baseline_minor, baseline_patch,
                    cuda_major, cuda_minor, cuda_patch,
                    cudnn_major, cudnn_minor, cudnn_patch,
                    python_major, python_minor, python_patch,
                    gpu_name, gpu_mem, cpu_name, cpu_mem, cpu_cores,
                    timestamp
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    STRFTIME('%s', 'now')
                );
                ''', (
                    speed['time'], speed['phase'], framework, dataset, model, task, config_str,
                    fw_v.major, fw_v.minor, fw_v.patch,
                    bl_v.major, bl_v.minor, bl_v.patch,
                    c_v.major, c_v.minor, c_v.patch,
                    nn_v.major, nn_v.minor, nn_v.patch,
                    p_v.major, p_v.minor, p_v.patch,
                    si['gpu_name'], si['gpu_mem'], si['cpu_name'], si['cpu_mem'], si['cpu_cores'],
                )
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise(e)
    finally:
        c.close()


def get_run_info(config):
    """Get task, dataset, and model info from a config."""
    task = config['task']
    dataset = config['dataset']
    model = config['model']['model_type']
    if task == 'classify':
        model = 'conv' if model == 'default' else model
        if model == 'lstm':
            if config['model'].get('rnn_type') == 'blstm':
                model = 'blstm'
    elif task == 'tagger':
        model = 'lstm' if model == 'default' else model
        if bool(config['model'].get('crf', True)):
            model = 'crf'
    elif task == 'seq2seq':
        model = 'vanilla' if model == 'default' else model
    elif task == 'lm':
        model = 'word' if model == 'default' else model
    return task, dataset, model


def get_framework_version(framework):
    """Get the version of a framework without importing them all at the same time."""
    if framework == "tensorflow":
        import tensorflow
        version = tensorflow.__version__
    elif framework == "pytorch":
        import torch
        version = torch.__version__
    elif framework == "dynet":
        import dynet
        version = dynet.__version__
    elif framework == "keras":
        import keras
        version = keras.__version__
    return version_str_to_tuple(version)


def version_str_to_tuple(version):
    m = simple_version_regex.match(version)
    if m is None:
        return Version(0, 0, 0)
    return Version(int(m.group(1)), int(m.group(2)), int(m.group(3)))

def get_cuda_version():
    """Find the version of CUDA and CUDNN."""
    cuda_version = open('/usr/local/cuda/version.txt').read().split()[-1]
    cudnn = open('/usr/local/cuda/include/cudnn.h').read()
    cudnn_major = re.search(r'CUDNN_MAJOR (\d)', cudnn).groups()[0]
    cudnn_minor = re.search(r'CUDNN_MINOR (\d)', cudnn).groups()[0]
    cudnn_patch = re.search(r'CUDNN_PATCHLEVEL (\d)', cudnn).groups()[0]
    cudnn_version = Version(cudnn_major, cudnn_minor, cudnn_patch)
    return version_str_to_tuple(cuda_version), cudnn_version


def get_gpu_info(gpu):
    """Get name and memory info about a GPU."""
    gpu_info = check_output('nvidia-smi --display=MEMORY -q --id={}'.format(gpu), shell=True).decode('utf-8')
    mem = re.search(r"^\s+Total\s+: (\d+ MiB)\s*$", gpu_info, re.M)
    gpu_mem = mem.groups()[0]
    gpu_mem = int(gpu_mem[:-3])

    gpu_dir = '/proc/driver/nvidia/gpus'
    gpu_file = sorted(os.listdir(gpu_dir))[int(gpu)]
    gpu_file = os.path.join(gpu_dir, gpu_file, 'information')
    gpu_name = open(gpu_file).read().split("\n")[0].split(":")[1].strip()
    return gpu_name, gpu_mem


def get_cpu_info():
    """Get CPU name, memory, and core count."""
    cpu_cores = cpu_count()

    cpu_info = open('/proc/cpuinfo').read().split("\n\n")[0]

    cpu_name = re.search(r"^model name\s+: (.*)$", cpu_info, re.M).groups()[0]

    mem_info = check_output('free -m', shell=True).decode('utf-8')
    mem_info = mem_info.split('\n')[1]
    cpu_mem = int(mem_info.split()[1])

    return cpu_name, cpu_mem, cpu_cores


def get_python_version():
    v = sys.version_info
    version = Version(v.major, v.minor, v.micro)
    return version


def edit_classify_config(config, frameworks=FRAMEWORKS):
    """Rotate frameworks."""
    configs = []
    for fw in frameworks:
        c = deepcopy(config)
        c['backend'] = fw
        configs.append(c)
    return configs


def edit_tagger_config(config, frameworks=FULL_FRAMEWORKS, no_crf=False):
    """Rotate frameworks and optionally remove the CRF."""
    configs = []
    for fw in frameworks:
        c = deepcopy(config)
        c['backend'] = fw
        if fw == 'dynet':
            # Dynet tagger only uses autobatching
            c['train']['autobatchsz'] = c['batchsz']
            c['batchsz'] = 1
        configs.append(c)
    if not no_crf:
        new_configs = []
        # Remove the CRF
        for config in configs:
            c = deepcopy(config)
            c['model']['crf'] = False
            c['model']['crf_mask'] = False
            new_configs.append(c)
            new_configs.append(config)
        configs = new_configs
    return configs


def edit_lm_config(config, frameworks=FULL_FRAMEWORKS):
    """Rotate frameworks."""
    configs = []
    for fw in frameworks:
        c = deepcopy(config)
        c['backend'] = fw
        configs.append(c)
    return configs


def edit_seq2seq_config(config, frameworks=FULL_FRAMEWORKS, no_attn=False):
    """Rotate frameworks and optionally remove attention."""
    configs = []
    for fw in frameworks:
        c = deepcopy(config)
        c['backend'] = fw
        configs.append(c)
    if not no_attn:
        new_configs = []
        # Run the non attention version
        for config in configs:
            c = deepcopy(config)
            c['model']['model_type'] = 'default'
            new_configs.append(c)
            new_configs.append(config)
        configs = new_configs
    return configs


def edit_config(config, frameworks=None, no_crf=False, no_attn=False):
    """Expand a config to test all the needed versions."""
    task = config['task']
    if task == 'classify':
        frameworks = frameworks if frameworks is not None else FRAMEWORKS
        return edit_classify_config(config, frameworks)
    if task == 'tagger':
        frameworks = frameworks if frameworks is not None else FULL_FRAMEWORKS
        return edit_tagger_config(config, frameworks, no_crf=no_crf)
    if task == 'lm':
        frameworks = frameworks if frameworks is not None else FULL_FRAMEWORKS
        return edit_lm_config(config, frameworks)
    if task == 'seq2seq':
        frameworks = frameworks if frameworks is not None else FULL_FRAMEWORKS
        return edit_seq2seq_config(config, frameworks, no_attn=no_attn)


def get_configs(path):
    """Get configs from disk, if it's a dir read all configs in it."""
    configs = []
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            configs.append(read_config_file(os.path.join(path, file_name)))
    else:
        configs.append(read_config_file(path))
    return configs


def run(args):
    # Limit it to a single GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    conn = create_db(args.db)
    m = Manager()

    logs = args.logging
    datasets = args.datasets
    embeddings = args.embeddings
    settings = args.settings

    # So we don't litter the fs
    dir_ = tempfile.mkdtemp(prefix='baseline-speed-test-')

    try:
        configs = get_configs(args.config)
        if not args.single:
            full_configs = []
            for config in configs:
                full_configs.extend(edit_config(config, args.frameworks, args.no_crf, args.no_attn))
            configs = full_configs
        if args.verbose:
            for config in configs:
                pprint(config)
                print()
            print()
        steps = len(configs) * args.trials
        pg = create_progress_bar(steps)
        for config in configs:
            for _ in range(args.trials):
                write_config = deepcopy(config)
                task_name = config['task']

                system_info = m.dict()
                p = Process(
                    target=run_model,
                    args=(
                        system_info,
                        config,
                        logs,
                        settings,
                        datasets,
                        embeddings,
                        task_name,
                        dir_,
                        int(args.gpu)
                    )
                )
                p.start()
                pid = p.pid
                p.join()
                # run_model(system_info, config, logs, settings, datasets, embeddings, task_name, dir_, int(args.gpu))
                # pid = os.getpid()
                log_file = os.path.join(dir_, 'reporting-{}.log'.format(pid))
                speeds = parse_logs(log_file)

                # Add dataset and model type to db?
                save_data(conn, speeds, write_config, system_info)
                pg.update()
        pg.done()
    finally:
        shutil.rmtree(dir_)


def add(args):
    conn = create_db(args.db)
    config = read_config_file(args.config)
    speeds = parse_logs(args.log)
    if not speeds:
        return
    si = {}
    si['framework_version'] = get_framework_version(config['backend'])
    si['cuda'], si['cudnn'] = get_cuda_version()
    si['gpu_name'], si['gpu_mem'] = get_gpu_info(args.gpu)
    si['cpu_name'], si['cpu_mem'], si['cpu_cores'] = get_cpu_info()
    si['python'] = get_python_version()
    si['baseline'] = version_str_to_tuple(baseline.__version__)

    save_data(conn, speeds, config, si)
