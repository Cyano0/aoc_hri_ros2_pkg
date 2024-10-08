#!/bin/bash

# It sets the shell options to:
# -e: Exit immediately if a command exits with a non-zero status.
# -x: Print commands and their arguments as they are executed.
set -e -x

rosdep update

sudo apt-get update 

cd ${WORKSPACES}/src
vcs import < ${WORKSPACES}/src/repos/external.repos
rosdep install --from-paths . -i -y


