import os
import re
import shutil
from setuptools import setup
from xpctl import __version__

class About(object):
    NAME = 'xpctl'
    AUTHOR = 'dpressel'
    VERSION = __version__
    EMAIL = "{}@gmail.com".format(AUTHOR)
    BASE_URL = "https://www.github.com/{}/baseline/tree/master".format(AUTHOR)
    URL = "{}/python/{}".format(BASE_URL, NAME)
    DOC_NAME = "docs/{}.md".format(NAME)
    DOC_URL = "{}/docs/".format(BASE_URL)

def fix_links(text):
    """Pypi doesn't seem to host multiple docs so replace local links with ones to github."""
    regex = re.compile(r"\[(.*?)\]\((.*?\.md)\)")
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
        name=About.NAME,
        version=About.VERSION,
        description='Experiment Control and Tracking',
        long_description=read_doc(About.DOC_NAME, "README.md"),
        long_description_content_type="text/markdown",
        author=About.AUTHOR,
        author_email=About.EMAIL,
        license='Apache 2.0',
        url=About.URL,
        packages=['xpctl'],
        install_requires=[
            'Click',
            'click-shell',
            'pymongo',
            'pandas',
            'xlsxwriter',
            'jsondiff'
        ],
        entry_points={
            'console_scripts': [
                'xpctl = xpctl.cli:cli'
            ],
        },
        extras_require={
            'test': []
        },
        classifiers={
            'Development Status :: 2 - Pre-Alpha',
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
        keywords=['experiment control', 'tracking'],
    )


if __name__ == "__main__":
    main()
