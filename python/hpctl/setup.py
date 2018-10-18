import os
import re
from setuptools import setup, find_packages


def get_version(project_name):
    regex = re.compile(r"^__version__ = '(\d+\.\d+\.\d+(?:a|b|rc|dev)?(?:\d)*?)'$")
    with open("{}/version.py".format(project_name)) as f:
        for line in f:
            m = regex.match(line)
            if m is not None:
                return m.groups(1)[0]


class About(object):
    NAME='hpctl'
    VERSION=get_version(NAME)
    AUTHOR=''
    EMAIL='{}@gmail.com'.format(AUTHOR)
    URL='https://github.com/{}/{}'.format(AUTHOR, NAME)
    DL_URL='{}/archive/{}.tar.gz'.format(URL, VERSION)
    LICENSE='Apache 2.0'
    DESCRIPTION='Automatic HyperParameter Optimization.'


def convert_images(text):
    image_regex = re.compile(r"!\[(.*?)\]\((.*?)\)")
    return image_regex.sub(r'<img src="\2" alt="\1">', text)

def collect_data(config_loc):
    """include everything in config_loc as package data."""
    configs = []
    for f in sorted(os.listdir(config_loc)):
        configs.append(os.path.join(config_loc, f))
    return configs


def write_manifest(lines):
    with open("MANIFEST.in", "w") as f:
        f.write("\n".join(map(lambda x: 'include {}'.format(x), lines)))


# def fix_links(text):
#     """Pypi doesn't seem to host multiple docs so replace local links with ones to github."""
#     regex = re.compile(r"\[(.*?)\]\(((?:docs|docker)/.*?\.md)\)")
#     text = regex.sub(r"[\1]({}\2)".format(About.DOC_URL), text)
#     return text


ext_modules = [
]

include_data = collect_data('hpctl/config')
include_data.extend(collect_data('hpctl/data'))
write_manifest(include_data)

setup(
    name=About.NAME,
    version=About.VERSION,
    description=About.DESCRIPTION,
    long_description=convert_images(open('README.md').read()),
    long_description_content_type="text/markdown",
    author=About.AUTHOR,
    author_email=About.EMAIL,
    url=About.URL,
    download_url=About.DL_URL,
    license=About.LICENSE,
    packages=find_packages(),
    package_data={
        'hpctl': include_data,
    },
    include_package_data=True,
    install_requires=[
        'six',
        'enum34',
    ],
    setup_requires=[
    ],
    extras_require={
        'test': ['pytest', 'mock'],
        'docker': ['docker'],
        'remote': ['flask', 'requests', 'cachetools']
    },
    keywords=['deep-learning', 'nlp', 'pytorch', 'tensorflow'],
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'hpctl = hpctl.main:main',
        ],
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
)
