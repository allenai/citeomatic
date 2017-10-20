#!/usr/bin/python
import setuptools

setuptools.setup(
    name='citeomatic',
    version='0.01',
    url='http://github.com/allenai/s2-research',
    packages=setuptools.find_packages(),
    install_requires=[
    ],
    tests_require=[
    ],
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
    pyrobuf_modules=['citeomatic.proto'],
)
