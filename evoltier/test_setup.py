from setuptools import setup, find_packages
import sys

sys.path.append('./model')
sys.path.append('./test')

setup(
    name='test',
    version='1.0',
    description='This is test codes for travis ci',
    packages=find_packages(),
    test_suite='make_test.suite'
)
