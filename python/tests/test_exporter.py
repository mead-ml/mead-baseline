import os
import glob
import shutil
from subprocess import call
import pytest
import numpy as np
tf = pytest.importorskip('tensorflow')

os.environ['CUDA_VISIBLE_DEVICES'] = ''

@pytest.fixture
def clean_up():
    loc = os.path.dirname(os.path.realpath(__file__))
    config_loc = os.path.join(loc, 'test_data', 'trained_class', 'sst2.json')
    model_loc = os.path.join(loc, 'test_data', 'trained_class', 'classify-model-tf-28762')
    save_loc = "{}.eval_saver".format(model_loc)
    output_loc = os.path.join(loc, 'models')
    yield loc, config_loc, model_loc, save_loc, output_loc
    for file_name in glob.glob('*.log'):
        try:
            os.remove(file_name)
        except:
            pass
    try:
        shutil.rmtree(output_loc)
    except:
        pass
    try:
        os.rename("{}.old".format(save_loc), save_loc)
    except:
        pass

def test_exporter_ema(clean_up):
    loc, config_loc, model_loc, save_loc, output_loc = clean_up
    EMA_VERSION = "1"
    NORM_VERSION = "2"

    cmd = "mead-export --config {config} --model {model_loc} --model_version {model_version} --output_dir {out}"
    call(cmd.format(config=config_loc, model_loc=model_loc, model_version=EMA_VERSION, out=output_loc), shell=True)
    os.rename(save_loc, "{}.old".format(save_loc))
    call(cmd.format(config=config_loc, model_loc=model_loc, model_version=NORM_VERSION, out=output_loc), shell=True)

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, ['serve'], os.path.join(output_loc, EMA_VERSION))
        norm_vars = [x for x in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not x.name.endswith('ExponentialMovingAverage')]
        ema_variables = sess.run(norm_vars)

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, ['serve'], os.path.join(output_loc, NORM_VERSION))
        norm_vars = [x for x in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not x.name.endswith('ExponentialMovingAverage')]
        ema_vars = [x for x in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if x.name.endswith('ExponentialMovingAverage')]
        norm_variables = sess.run(norm_vars)
        ema_values = sess.run(ema_vars)

    for v1, v2 in zip(ema_variables, ema_values):
        np.testing.assert_allclose(v1, v2)

    for v1, v2 in zip(ema_variables, norm_variables):
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(v1, v2)
