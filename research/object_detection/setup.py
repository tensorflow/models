"""Setup script for object_detection."""

import sys
import os
import subprocess
import shutil

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1']
packages = [p for p in find_packages() if p.startswith('object_detection')]


def protobuf_compiler():
    print("Compiling Protobuf files...")

    protoc_binary = shutil.which("protoc")
    pb_dir_path = os.path.join("object_detection", "protos")

    for pb_filename in os.listdir(pb_dir_path):
        if pb_filename.endswith(".proto"):

            protoc_command = [protoc_binary, os.path.join(pb_dir_path, pb_filename), "--python_out=."]
            if subprocess.call(protoc_command) != 0:
                print("**ERROR during Protobuf files compilation: {}**".format(pb_filename))
                sys.exit(1)

    print("Protobuf files Compilation Done.")


class CustomInstallCommand(install):
    def run(self):
        protobuf_compiler()
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        protobuf_compiler()
        develop.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        protobuf_compiler()
        egg_info.run(self)


setup(name='object_detection',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=packages,
      cmdclass={"install": CustomInstallCommand,
                "develop": CustomDevelopCommand,
                "egg_info": CustomEggInfoCommand},
      description='Tensorflow Object Detection Library'
      )
