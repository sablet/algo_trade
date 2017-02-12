# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as test_command


class PyTest(test_command):

    user_options = [
        ('pytest-args=', 'a', 'Arguments for pytest'),
    ]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_target = []
        self.pytest_args = []

    def finalize_options(self):
        test_command.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

version = '0.1.0'


setup_requires = [
    'pytest'
]
tests_require = [
    'pytest-timeout',
    'mypy-lang',
]

setup(
    name='algo_trade',
    package=find_packages(),
    setup_requires=setup_requires,
    ## install_requires=install_requires,
    tests_require=tests_require,
    cmdclass={'test': PyTest},
    test_suite='test'
)
