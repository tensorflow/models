#!/bin/bash

# The current directory
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================"
echo "Setup .bashrc"
echo "================================================================"
# Shell config files that various shells source when they run.
# This is where we want to add aliases, source ROS environment
# variables, etc.
SHELL_CONFIG_FILES=(
    "$HOME/.bashrc"\
            "$HOME/.zshrc"
    )

# All lines listed here will be added to the shell config files
# listed above, if they are not present already
declare -a new_shell_config_lines=(
    # Make sure that all shells know where to find our custom gazebo models,
    # plugins, and resources. Make sure to preserve the path that already exists as well
    export PYTHONPATH=$PYTHONPATH:$CURR_DIR/research:$CURR_DIR/research/slim
)

#Installing python dependencies
pip install virtualenv

virtualenv subbots_python
source $CURR_DIR/subbots_python/bin/activate
pip install -r $CURR_DIR/requirements.txt

source $HOME/.bashrc

#Testing installation
python $CURR_DIR/research/object_detection/builders/model_builder_test.py



