import os
import shutil
import argparse
from subprocess import call
import setuptools

# Only newer setuptools allow for you to upload markdown to PyPi
SETUP_MAJOR = 39
SETUP_MINOR = 1
SETUP_PATCH = 0
SETUP_ERROR = (
    "Please update setuptools: required {r_major}.{r_minor}.{r_patch}, "
    "found {f_major}.{f_minor}.{f_patch}"
)

major, minor, patch = map(int, setuptools.__version__.split("."))
versions = {
    "r_major": SETUP_MAJOR,
    "r_minor": SETUP_MINOR,
    "r_patch": SETUP_PATCH,
    "f_major": major,
    "f_minor": minor,
    "f_patch": patch,
}

assert major >= SETUP_MAJOR, SETUP_ERROR.format(**versions)
if major == SETUP_MAJOR:
    assert minor >= SETUP_MINOR, SETUP_ERROR.format(**versions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", choices=['baseline', 'xpctl'])
    args = parser.parse_args()

    file_name = "setup_{}.py".format(args.package)

    try:
        shutil.copyfile(file_name, "setup.py")
        call("python setup.py sdist", shell=True)

    finally:
        try:
            os.remove('setup.py')
        except OSError:
            pass
        try:
            os.remove('README.md')
        except OSError:
            pass
        try:
            os.remove('MANIFEST.in')
        except OSError:
            pass

if __name__ == "__main__":
    main()
