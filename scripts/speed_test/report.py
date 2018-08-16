from __future__ import print_function
from six.moves import shlex_quote

import os
import shutil
import sqlite3
import tempfile
from subprocess import check_call, call
from collections import defaultdict
from itertools import repeat, chain
import numpy as np
import mead
from mead.utils import convert_path
from baseline.utils import read_config_file


class Version(object):
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch
    def __str__(self):
        return ".".join(map(str, [self.major, self.minor, self.patch]))


def explore(args):
    if not os.path.exists(args.db):
        print("Unable to find database `{}`".format(args.db))
        return
    conn = sqlite3.connect(args.db)
    create_results(conn)
    call(['sqlite3', '-column', '-header', shlex_quote(args.db)])


def query(args):
    if not os.path.exists(args.db):
        print("Unable to find database `{}`".format(args.db))
        return
    conn = sqlite3.connect(args.db)
    task = args.task
    dataset = args.dataset
    frameworks = args.frameworks if args.frameworks is not None else get_frameworks(conn, task, dataset)
    models = args.models if args.models is not None else get_models(conn, task, dataset)
    create_results(conn)
    results = defaultdict(list)
    for fw in frameworks:
        for model in models:
            m, s, t, d, f, ms = get_results_by_version(
                conn, task, dataset, fw, model, 'Train',
            )
            results['framework'].extend(f)
            results['model'].extend(ms)
            results['task'].extend(t)
            results['dataset'].extend(d)
            results['mean'].extend(m)
            results['std'].extend(s)
    keys = results.keys()
    for k in keys:
        if k == 'mean' or k == 'std':
            print(" {:>10} ".format(k), end='')
        else:
            print(" {:<10} ".format(k), end='')
    print()
    for k in keys:
        print(" " + "-" * 10 + " ", end='')
    print()
    for i in range(len(results['mean'])):
        for k in keys:
            if isinstance(results[k][i], float):
                print(" {:=10.5f} ".format(results[k][i]), end='')
            else:
                print(" {:<10} ".format(results[k][i]), end='')
        print()


PREAMBLE = r'''
\documentclass{standalone}
\usepackage{multirow}
\begin{document}
'''

END = r'''
\end{document}
'''

MARKDOWN = '''
# Speed Tests

This is a speed test of different baseline models and frameworks.

'''


results_query = '''
CREATE TABLE results AS
SELECT
    sub.runs as runs,
    sub.latest as latest,
    sub.mean as mean,
    SUM((speed.time - sub.mean) * (speed.time - sub.mean)) / (COUNT(1) - 1) as var,
    speed.task as task,
    speed.dataset as dataset,
    speed.model as model,
    speed.framework as framework,
    speed.phase as phase,
    speed.framework_major as framework_major,
    speed.framework_minor as framework_minor,
    speed.framework_patch as framework_patch,
    speed.baseline_major as baseline_major,
    speed.baseline_minor as baseline_minor,
    speed.baseline_patch as baseline_patch,
    speed.cuda_major as cuda_major,
    speed.cuda_minor as cuda_minor,
    speed.cuda_patch as cuda_patch,
    speed.cudnn_major as cudnn_major,
    speed.cudnn_minor as cudnn_minor,
    speed.cudnn_patch as cudnn_patch,
    speed.python_major as python_major,
    speed.python_minor as python_minor,
    speed.python_patch as python_patch,
    speed.gpu_name, speed.gpu_mem,
    speed.cpu_name, speed.cpu_mem, speed.cpu_cores,
    speed.config
FROM speed JOIN (
    SELECT
        MAX(timestamp) as latest,
        COUNT(time) as runs,
        AVG(time) as mean,
        task, framework, dataset, model, phase,
        framework_major, framework_minor, framework_patch,
        baseline_major, baseline_minor, baseline_patch,
        cuda_major, cuda_minor, cuda_patch,
        cudnn_major, cudnn_minor, cudnn_patch,
        python_major, python_minor, python_patch,
        gpu_name, gpu_mem,
        cpu_name, cpu_mem, cpu_cores,
        config
    FROM speed GROUP BY
        phase, framework, dataset, model, task,
        framework_major, framework_minor, framework_patch,
        baseline_major, baseline_minor, baseline_patch,
        cuda_major, cuda_minor, cuda_patch,
        cudnn_major, cudnn_minor, cudnn_patch,
        python_major, python_minor, python_patch,
        gpu_name, gpu_mem,
        cpu_name, cpu_mem, cpu_cores,
        config
) AS sub ON
    speed.task = sub.task AND
    speed.framework = sub.framework AND
    speed.dataset = sub.dataset AND
    speed.model = sub.model AND
    speed.phase = sub.phase AND
    speed.framework_major = sub.framework_major AND
    speed.framework_minor = sub.framework_minor AND
    speed.framework_patch = sub.framework_patch AND
    speed.baseline_major = sub.baseline_major AND
    speed.baseline_minor = sub.baseline_minor AND
    speed.baseline_patch = sub.baseline_patch AND
    speed.cuda_major = sub.cuda_major AND
    speed.cuda_minor = sub.cuda_minor AND
    speed.cuda_patch = sub.cuda_patch AND
    speed.cudnn_major = sub.cudnn_major AND
    speed.cudnn_minor = sub.cudnn_minor AND
    speed.cudnn_patch = sub.cudnn_patch AND
    speed.python_major = sub.python_major AND
    speed.python_minor = sub.python_minor AND
    speed.python_patch = sub.python_patch AND
    speed.gpu_name = sub.gpu_name AND speed.gpu_mem = sub.gpu_mem AND
    speed.cpu_name = sub.cpu_name AND speed.cpu_mem = sub.cpu_mem AND speed.cpu_cores = sub.cpu_cores AND
    speed.config = sub.config
GROUP BY
    speed.phase,
    speed.framework,
    speed.dataset,
    speed.model,
    speed.task,
    speed.framework_major,
    speed.framework_minor,
    speed.framework_patch,
    speed.baseline_major,
    speed.baseline_minor,
    speed.baseline_patch,
    speed.cuda_major,
    speed.cuda_minor,
    speed.cuda_patch,
    speed.cudnn_major,
    speed.cudnn_minor,
    speed.cudnn_patch,
    speed.python_major,
    speed.python_minor,
    speed.python_patch,
    speed.gpu_name, speed.gpu_mem,
    speed.cpu_name, speed.cpu_mem, speed.cpu_cores,
    speed.config
HAVING runs >= 2
ORDER BY speed.task, speed.dataset, speed.model, speed.framework, speed.phase,
framework_major, framework_minor, framework_patch,
baseline_major, baseline_minor, baseline_patch,
cuda_major, cuda_minor, cuda_patch,
cudnn_major, cudnn_minor, cudnn_patch,
python_major, python_minor, python_patch,
latest;
'''


def create_results(conn):
    try:
        c = conn.cursor()
        c.execute('''
            DROP TABLE results;
        ''')
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
    finally:
        c.close()
    try:
        c = conn.cursor()
        c.execute(results_query)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise(e)
    finally:
        c.close()


def get_tasks(conn):
    tasks = conn.execute('''
        SELECT DISTINCT task FROM speed ORDER BY task;
    ''').fetchall()
    return [t[0] for t in tasks]


def get_datasets(conn, task):
    datasets = conn.execute('''
        SELECT DISTINCT dataset FROM speed WHERE task = ? ORDER BY dataset;
    ''', (task,)
    ).fetchall()
    return [d[0] for d in datasets]


def get_frameworks(conn, task, dataset):
    frameworks = conn.execute('''
        SELECT DISTINCT framework FROM speed
        WHERE task = ? AND dataset = ?
        ORDER BY framework;
        ''', (task, dataset)
    ).fetchall()
    return [f[0] for f in frameworks]


def get_models(conn, task, dataset):
    models = conn.execute('''
        SELECT DISTINCT model FROM speed
        WHERE task = ? AND dataset = ?
        ORDER BY model;
        ''', (task, dataset)
    ).fetchall()
    return [m[0] for m in models]


def get_phases(conn, task, dataset):
    phases = conn.execute('''
        SELECT DISTINCT phase FROM speed
        WHERE task = ? AND dataset = ?
        ORDER BY phase;
        ''', (task, dataset)
    ).fetchall()
    phases = {p[0] for p in phases}
    return ['Train', 'Valid', 'Test']


def get_results(conn, task, dataset, framework, model, phase):
    mean, var = conn.execute('''
        SELECT mean, var FROM results
        WHERE task = ? AND dataset = ? AND framework = ? AND model = ? AND phase = ?
        ORDER BY
            framework_major DESC, framework_minor DESC, framework_patch DESC,
            baseline_major DESC, baseline_minor DESC, baseline_patch DESC,
            cuda_major DESC, cuda_minor DESC, cuda_patch DESC,
            cudnn_major DESC, cudnn_minor DESC, cudnn_patch DESC,
            python_major DESC, python_minor DESC, python_patch DESC
        LIMIT 1;
        ''',
        (task, dataset, framework, model, phase)
    ).fetchall()[0]
    return mean, np.sqrt(var)


def get_results_by_version(conn, task, dataset, framework, model, phase):
    results = conn.execute('''
        SELECT
            mean, var, task, dataset, framework, model
        FROM results
        WHERE task = ? AND dataset = ? AND framework = ? AND model = ? AND phase = ?
        ORDER BY
            framework_major DESC, framework_minor DESC, framework_patch DESC,
            baseline_major DESC, baseline_minor DESC, baseline_patch DESC,
            cuda_major DESC, cuda_minor DESC, cuda_patch DESC,
            cudnn_major DESC, cudnn_minor DESC, cudnn_patch DESC,
            python_major DESC, python_minor DESC, python_patch DESC
        LIMIT 1;
        ''',
        (task, dataset, framework, model, phase)
    ).fetchall()
    mean = [r[0] for r in results]
    std = [np.sqrt(r[1]) for r in results]
    task = [r[2] for r in results]
    dataset = [r[3] for r in results]
    framework = [r[4] for r in results]
    model = [r[5] for r in results]
    return mean, std, task, dataset, framework, model


def get_env(conn, task, dataset, framework, model, phase):
    env = conn.execute('''
        SELECT
            framework_major, framework_minor, framework_patch,
            baseline_major, baseline_minor, baseline_patch,
            cuda_major, cuda_minor, cuda_patch,
            cudnn_major, cudnn_minor, cudnn_patch,
            python_major, python_minor, python_patch,
            gpu_name, gpu_mem,
            cpu_name, cpu_mem, cpu_cores
        FROM results
        WHERE task = ? AND dataset = ? AND framework = ? AND model = ? AND phase = ?
        ORDER BY
            framework_major DESC, framework_minor DESC, framework_patch DESC,
            baseline_major DESC, baseline_minor DESC, baseline_patch DESC,
            cuda_major DESC, cuda_minor DESC, cuda_patch DESC,
            cudnn_major DESC, cudnn_minor DESC, cudnn_patch DESC,
            python_major DESC, python_minor DESC, python_patch DESC
        LIMIT 1;
        ''',
        (task, dataset, framework, model, phase)
    ).fetchall()[0]
    _env = {}
    _env['framework'] = Version(env[0], env[1], env[2])
    _env['baseline'] = Version(env[3], env[4], env[5])
    _env['cuda'] = Version(env[6], env[7], env[8])
    _env['cudnn'] = Version(env[9], env[10], env[11])
    _env['python'] = Version(env[12], env[13], env[14])
    _env['gpu_name'] = env[15]
    _env['gpu_mem'] = env[16]
    _env['cpu_name'] = env[17]
    _env['cpu_mem'] = env[18]
    _env['cpu_cores'] = env[19]
    return _env


def create_table(out, task, table):
    dir_name = "images_{}".format(out)
    tmp = tempfile.mkdtemp()
    curr = os.getcwd()
    os.chdir(tmp)
    pdf = create_latex(task, table)
    pic = create_png(pdf)
    pic_file = os.path.join(tmp, pic)
    os.chdir(curr)
    try:
        os.mkdir(dir_name)
    except:
        pass
    pic = os.path.join(dir_name, pic)
    os.rename(pic_file, pic)
    shutil.rmtree(tmp)
    return pic


def create_latex(file_name, table):
    latex = "\n".join([PREAMBLE, table, END])
    with open(file_name, 'w') as f:
        f.write(latex)
    res = check_call('pdflatex {}'.format(file_name), shell=True, stdout=open(os.devnull, 'w'))
    return file_name + '.pdf'


def create_png(file_name):
    pic = file_name[:-3] + 'png'
    res = check_call('convert -density 300 {} -quality 90 {}'.format(file_name, pic), shell=True)
    res = check_call('convert {0} -background white -alpha remove {0}'.format(pic), shell=True)
    return pic


def build_table(conn, task, dataset, models, frameworks, phases):
    cols = ('c' * len(frameworks) + "|")
    table_start = r'\begin{tabular}{|c|c|%s|}\hline' % cols

    table_header = ['\multicolumn{2}{|c|}{%s}' % dataset]
    for fw in frameworks:
        table_header.append(fw)
    table_header = "&".join(table_header) + r"\\ \hline"

    rows = []
    for phase in phases:
        label = '\multirow{%d}{*}{%s}' % (len(models), phase)
        first = True
        for model in models:
            if first:
                model_row = [phase]
                first = False
            else:
                model_row = [' ']
            model_row.append(model)
            for framework in frameworks:
                mean, std = get_results(conn, task, dataset, framework, model, phase)
                model_row.append("${:.2f} \pm {:.2f}$".format(mean, std))
            rows.append("&".join(model_row) + r"\\ \cline{2-2}")
        rows.append("\hline")
    table_end = "\hline\n\end{tabular}"
    x = "\n".join([
        table_start,
        table_header,
        *rows,
        table_end
    ])
    return x


def create_envs(conn, task, dataset, models, frameworks):
    md = ['\n\n<details><summary>Environment</summary>\n<p>\nThese results were runs on the following configurations:\n\n']
    for framework in frameworks:
        for model in models:
            env = get_env(conn, task, dataset, framework, model, 'Train')
            md.append(' * {} - {}'.format(framework, model))
            for k, v in env.items():
                md.append('    * {} - {}'.format(k, v))
    md.append("\n\n</p>\n</details>\n\n")
    return md


def report(args):
    if not os.path.exists(args.db):
        print("Unable to find database `{}`".format(args.db))
        return
    conn = sqlite3.connect(args.db)
    create_results(conn)
    tasks = get_tasks(conn)
    markdown = [MARKDOWN]
    for task in tasks:
        datasets = get_datasets(conn, task)
        for dataset in datasets:
            frameworks = get_frameworks(conn, task, dataset)
            models = get_models(conn, task, dataset)
            phases = get_phases(conn, task, dataset)
            table = build_table(conn, task, dataset, models, frameworks, phases)
            pic = create_table(args.out, task, table)
            markdown.append("\n## {}".format(task))
            markdown.append("")
            markdown.append("![](./{})".format(pic))
            markdown.append("")
            envs = create_envs(conn, task, dataset, models, frameworks)
            markdown.extend(envs)
    with open(args.out + ".md", 'w') as f:
        f.write("\n".join(markdown))
