#!bin/bash
#
# Script that installs the requirements for custom
# git precommit hook and creates symbolic link between
# hook in the repository and local git hook

printf ">>> Installing required python packages...\n"
python3 -m pip install -r requirements.txt

printf "\n>>> Creating symbolic link between pre-commit file and local git pre-commit hook..."
ln -s -f ../../hooks/pre-commit .git/hooks/pre-commit

printf "\n\nInstallation complete."
