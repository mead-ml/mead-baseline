import ast
from setuptools import setup, find_packages


def get_version(file_name, version_name="__version__"):
    with open(file_name) as f:
        tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_name:
                    return node.value.s
    raise ValueError(f"Unable to find an assignment to the variable {version_name} in file {file_name}")


class About(object):
    NAME = "mead-layers"
    AUTHOR = "mead-ml"
    VERSION = get_version("eight_mile/version.py")
    EMAIL = "mead.baseline@gmail.com"
    URL = "https://www.github.com/{}/{}".format(AUTHOR, NAME)
    DOWNLOAD_URL = "{}/archive/{}.tar.gz".format(URL, VERSION)
    LICENSE = "Apache 2.0"
    DESCRIPTION = "Reusable Deep-Learning layers for NLP"


def main():
    setup(
        name=About.NAME,
        version=About.VERSION,
        description=About.DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author=About.AUTHOR,
        author_email=About.EMAIL,
        license=About.LICENSE,
        url=About.URL,
        download_url=About.DOWNLOAD_URL,
        packages=find_packages(exclude=["tests"]),
        include_package_data=False,
        install_requires=["numpy"],
        extras_require={
            "test": ["pytest", "mock", "contextdecorator", "pytest-forked"],
            "yaml": ["pyyaml"],
            "tf2": ["tensorflow_addons"],
            "plot": ["matplotlib"],
        },
        entry_points={"console_scripts": ["bleu = eight_mile.bleu:main" "conlleval = eight_mile.conlleval:main"]},
        classifiers={
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        },
        keywords=["deep-learning", "nlp", "pytorch", "tensorflow"],
    )


if __name__ == "__main__":
    main()
