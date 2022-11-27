#!/usr/bin/env python
# coding: utf-8

from setuptools import setup
from report_manager.version import __version__, __author__
from pathlib import Path
import re


readme = Path('README.md').read_text()
version = __version__
readme = re.sub(r"Version: [0-9]*\.[0-9]*\.[0-9][0-9]*", f"Version: {version}", readme)
print(f"README: {readme}")
with open("README.md", "w") as readme_file:
    readme_file.write(readme)
description = 'Manage your reports'

setup(
    name='report-manager',
    version=f"{version}",
    description=description,
    long_description=f'{readme}',
    long_description_content_type='text/markdown',
    url='https://github.com/Knuckles-Team/report-manager',
    author=__author__,
    author_email='knucklessg1@gmail.com',
    license='Unlicense',
    packages=['report_manager'],
    include_package_data=True,
    install_requires=['scikit-learn', 'pandas', 'matplotlib', 'pandas_profiling', 'scipy', 'numpy'],
    py_modules=['report_manager'],
    package_data={'report_manager': ['report_manager']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: Public Domain',
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={'console_scripts': ['report-manager = report_manager.report_manager:main']},
)
