# syntax = devthefuture/dockerfile-x

ARG BASE_IMAGE=ros:humble

############################################################################################################
FROM ${BASE_IMAGE} as base

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Set the LD_PRELOAD environment variable
ENV LD_PRELOAD=/usr/lib/libdlfaker.so:/usr/lib/libvglfaker.so

# Define the workspace path
ENV WORKSPACES=/workspaces/aoc_hri_ros2_pkg

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    python3-pip \
    python3-rosdep


# Install all required APT packages
INCLUDE .devcontainer/user_install.dockerfile

############################################################################################################
# get the source tree and analyse it for its package.xml only
FROM base as sourcefilter
COPY ./src /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

############################################################################################################
# install all dependencies listed in the package.xml
FROM base as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

############################################################################################################
FROM depbuilder as final

# Copy the full src directory into the final stage
COPY ./src ${WORKSPACES}/src

# add user ros with sudo rights if it doesn't exist
RUN if ! id ros; then \
        useradd -ms /bin/bash ros && echo "ros:ros" | chpasswd && adduser ros sudo; \
        echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers; \
        echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc; \
    else echo "user ros already exists"; \
    fi

# add sudo without password
ENV DEBIAN_FRONTEND=noninteractive

USER ros