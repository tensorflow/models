# About 
This folder contains a pre-commit hook that runs the following 
modules in bash when a user types "git commit":
- isort
- mypy
- pylint.sh 

# Dependencies 
You will need to have the following installed on your machine:
- isort 
- mypy 
- pylint

They will be installed through the ```install.sh``` script, or you can 
install them manually.

# Setup
## Required Files 
You will need to have the following in the root directory of your project:
- ```hooks``` directory, which contains the ```pre-commit``` hook script
- ```pylintrc``` config file
 
The files are located [here](https://github.com/PurdueDualityLab/tf-models/tree/mesh_rcnn_linter)
in the root folder of the ```mesh_rcnn_linter``` branch:

![folder_structure](./screenshots/folder_structure.png)

## install.sh
You can easily set everything up by navigating into the ```hooks``` folder and running the 
install script with the following command in your terminal: ```bash install.sh```. 

__NOTE__: make sure your ```install.sh``` file is set up as an executable; if it isn't, run
the command ```chmod +x install.sh```. 

## Manual Installation 
If you want to set things up manually, what you can do is the following:
- copy and paste the ```pre-commit``` script into your local ```.git/hooks``` directory like so:

``` mv hooks/precommit .git/hooks```

OR 
- create a symbolic link from the ```hooks``` directory to the ```.git/hooks``` directory
using the following command (run the command in project root directory):

```ln -s -f ../../hooks/pre-commit .git/hooks/pre-commit```

## How to use the linter 
After running ```git add .```, when you run ```git commit``` on your local machine, the pre-commit script
will run the linter to scan the ```$PROJECT_DIR``` defined in ```pre-commit```.

If there are any formatting issues with your code, you should see something like this:

![failed_commit](./screenshots/failed_commit.png)

and your changes will not be commited.

After you correct any formatting issues, run ```git commit``` again and your changes
will be committed successfully and you should see a message like this:

![passed_commit](./screenshots/passed_commit.png)

You will now be able to push your local changes.

# Important Notes
- You need to make sure the ```pre-commit``` file is an 
executable; use the command ```sudo chmod +x pre-commit``` to make it an executable
- If you run into permission issues with "git add . " command when running the pre-commit hook, 
use the following command to fix the issue: ```sudo chown -R "${USER:-$(id -un)}" . ```
