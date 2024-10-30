# syntax = devthefuture/dockerfile-x

ARG BASE_IMAGE=lcas.lincoln.ac.uk/lcas/ros:jammy-humble-cuda12.2-opengl
############################################################################################################
FROM ${BASE_IMAGE} as base

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ARG user_id=1000
ENV user_id=${user_id}

ARG COLCON_WS=/home/ros/aoc_hri_ws
ENV COLCON_WS=${COLCON_WS}

# Install all required APT packages
INCLUDE .devcontainer/docker_include/apt_install.dockerfile

# Include env variables 
INCLUDE .devcontainer/docker_include/env_setup.dockerfile

############################################################################################################
FROM base AS user_space

INCLUDE .devcontainer/docker_include/user_install.dockerfile

############################################################################################################
# get the source tree and analyse it for its package.xml only
FROM user_space as sourcefilter
COPY . /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

############################################################################################################
# install all dependencies listed in the package.xml
FROM user_space as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

############################################################################################################
FROM depbuilder as final

# add user ros with sudo rights if it doesn't exist
# RUN if ! id ros; then \
#         useradd -ms /bin/bash ros && echo "ros:ros" | chpasswd && adduser ros sudo; \
#         echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers; \
#         echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc; \
#     else echo "user ros already exists"; \
#     fi

# ENV DEBIAN_FRONTEND=noninteractive

USER $USER


# ARG AOC_HRI_PKG_DIR=${COLCON_WS}/src/aoc_hri_ros2_pkg
# ENV AOC_HRI_PKG_DIR=${AOC_HRI_PKG_DIR}

# RUN chown -R ${USER}:${USER} ${COLCON_WS}

# copy entire COLCON_WS
# Copies the contents of the current directory to the specified service directory
# within the container, setting the ownership to the user specified by the $USER
# environment variable.
# COPY --chown=$USER . ${AOC_HRI_PKG_DIR}

CMD ["/bin/bash"]