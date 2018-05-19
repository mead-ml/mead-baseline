import os
import re
import shutil
from itertools import product
from setuptools import setup
from baseline import __version__

class About(object):
    NAME = 'baseline'
    AUTHOR = 'dpressel'
    VERSION = __version__
    EMAIL = "{}@gmail.com".format(AUTHOR)
    URL = "https://www.github.com/{}/{}".format(AUTHOR, NAME)
    DOWNLOAD_URL = "{}/archive/{}.tar.gz".format(URL, VERSION)
    DOC_URL = "{}/tree/master/".format(URL)
    DOC_NAME = 'README.md'

def build_packages(backends, tasks):
    """Specify backends by hand so dev ones aren't included."""
    packages = product(backends, tasks)
    packages = list(map(lambda x: 'baseline.{}'.format( '.'.join(x)), packages))
    packages += list(map(lambda x: 'baseline.{}'.format(x), backends))
    return packages

def get_configs(config_loc):
    """include everything in config_loc as package data."""
    configs = []
    for f in os.listdir(config_loc):
        configs.append(os.path.join(config_loc, f))
    write_manifest(configs)
    return configs

def write_manifest(lines):
    with open("MANIFEST.in", "w") as f:
        f.write("\n".join(map(lambda x: 'include {}'.format(x), lines)))

def fix_links(text):
    """Pypi doesn't seem to host multiple docs so replace local links with ones to github."""
    regex = re.compile(r"\[(.*?)\]\(((?:docs|docker)/.*?\.md)\)")
    text = regex.sub(r"[\1]({}\2)".format(About.DOC_URL), text)
    return text

def read_doc(f_name, new_name=None, fix_fn=fix_links):
    """
    Because our readme is outside of this dir we need to copy it in so
    that it is picked up by the install.
    """
    if new_name is None:
        new_name = f_name
    path = os.path.dirname(os.path.realpath(__file__))
    doc_loc = os.path.join(path, '..', f_name)
    new_loc = os.path.join(path, new_name)
    if os.path.isfile(doc_loc):
        shutil.copyfile(doc_loc, new_loc)
    descript = open(new_loc, 'r').read()
    return fix_fn(descript)

def main():
    setup(
        name='deep_{}'.format(About.NAME),
        version=About.VERSION,
        description='Strong Deep-Learning Baseline algorithms for NLP',
        long_description=read_doc(About.DOC_NAME),
        long_description_content_type="text/markdown",
        author=About.AUTHOR,
        author_email=About.EMAIL,
        license='Apache 2.0',
        url=About.URL,
        download_url=About.DOWNLOAD_URL,
        packages=(
            ['baseline', 'mead', 'mead.tf', 'addons'] +
            build_packages(['pytorch', 'tf'], ['classify', 'seq2seq', 'lm', 'tagger']) +
            build_packages(['keras'], ['classify'])
        ),
        package_data={
            'mead': get_configs('mead/config'),
        },
        include_package_data=True,
        install_requires=[
            'numpy',
            'six',
            'requests',
        ],
        extras_require={
            'test': ['pytest', 'mock']
        },
        entry_points={
            'console_scripts': [
                'mead-train = mead.trainer:main',
                'mead-export = mead.export:main',
            ]
        },
        classifiers={
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        },
        keywords=['deep-learning', 'nlp', 'pytorch', 'tensorflow'],
    )

if __name__ == "__main__":
    main()
