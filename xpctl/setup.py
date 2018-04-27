from setuptools import setup

setup(
    name="xpctl",
    version='0.1',
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
          ]
      },
)
