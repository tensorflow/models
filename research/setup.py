"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup
import os
import sys

REQUIRED_PACKAGES = ['Pillow>=1.0']

def compile_protos():
    absolute_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(absolute_path)
    os.chdir(script_dir)

    cmd = 'protoc object_detection/protos/*.proto --python_out=.'
    if os.system(cmd) != 0:
        sys.exit(-1)

if __name__ == "__main__":
    compile_protos()
    setup(
        name='object_detection',
        version='0.1',
        install_requires=REQUIRED_PACKAGES,
        include_package_data=True,
        packages=[p for p in find_packages() if p.startswith('object_detection')],
        description='Tensorflow Object Detection Library',
    )
