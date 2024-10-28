#!/bin/bash

# Exit immediately if a command exits with a non-zero status and enable debug mode for verbose output
set -e -x

# Update rosdep database to ensure all dependencies are up-to-date
echo "Running rosdep update..."
rosdep update

# Update the list of available packages and versions in apt-get
echo "Updating apt-get packages..."
apt-get update

# Navigate to the src directory of the current workspace
cd "$(pwd)/src"

# Import repositories using vcs
echo "Importing repositories with vcs..."
vcs import --force --recursive < ../.devcontainer/repos/external.repos 

# Install any dependencies specified in the cloned repositories' package.xml files
# The '-r' flag makes the process recursive
# The '-y' flag automatically answers "yes" to any prompts
rosdep install -r -y -i --from-paths .

# Install Python dependencies for the ultralytics_ros package
python3 -m pip install -r ultralytics_ros/requirements.txt

echo "Setup complete."
