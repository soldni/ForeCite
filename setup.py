#!/usr/bin/python

import setuptools
from pathlib import Path

requirements = Path(__file__).parent / 'requirements.txt'
with open(requirements) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="ForeCite",
    version="0.0.1",
    url="https://github.com/allenai/ForeCite",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    tests_require=[],
    zip_safe=False,
    test_suite="py.test",
    entry_points="",
)
