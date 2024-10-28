#!/bin/bash

set -e -x

echo "Running rosdep update..."
rosdep update

echo "Updating apt-get packages..."
apt-get update

# Navigate to the src directory
cd "$(pwd)/src"

# Import repositories using vcs
echo "Importing repositories with vcs..."
vcs import --force --recursive < ../.devcontainer/repos/external.repos 

# Install dependencies
echo "Installing dependencies with rosdep..."
rosdep install --from-paths . -i -y

echo "Setup complete."