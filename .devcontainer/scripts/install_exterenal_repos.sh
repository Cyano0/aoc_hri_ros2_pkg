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

if [ ! -d "ultralytics_ros" ]; then
    echo "Cloning ultralytics_ros repository..."
    GIT_LFS_SKIP_SMUDGE=1 git clone -b humble-devel https://github.com/LCAS/ultralytics_ros.git
else
    echo "ultralytics_ros directory already exists, skipping clone."
fi

# Install any dependencies specified in the cloned repositories' package.xml files
# The '-r' flag makes the process recursive
# The '-y' flag automatically answers "yes" to any prompts
rosdep install -r -y -i --from-paths .

# Install Python dependencies for the ultralytics_ros package
python3 -m pip install -r ultralytics_ros/requirements.txt

echo "Setup complete."
