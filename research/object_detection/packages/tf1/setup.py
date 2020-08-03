"""Setup script for object_detection with TF1.0."""
import os
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pillow', 'lxml', 'matplotlib', 'Cython',
                     'contextlib2', 'tf-slim', 'six', 'pycocotools', 'scipy',
                     'pandas']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=(
        [p for p in find_packages() if p.startswith('object_detection')] +
        find_packages(where=os.path.join('.', 'slim'))),
    package_dir={
        'datasets': os.path.join('slim', 'datasets'),
        'nets': os.path.join('slim', 'nets'),
        'preprocessing': os.path.join('slim', 'preprocessing'),
        'deployment': os.path.join('slim', 'deployment'),
        'scripts': os.path.join('slim', 'scripts'),
    },
    description='Tensorflow Object Detection Library with TF1.0',
    python_requires='>3.6',
)
