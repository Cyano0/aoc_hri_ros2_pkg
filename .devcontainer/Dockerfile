ARG BASE_IMAGE=ros:humble

FROM ${BASE_IMAGE} as base

USER root

ENV DEBIAN_FRONTEND=noninteractive

ENV WORKSPACES=/workspaces/aoc_hri_ros2_pkg/src

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    python3-pip \
    python3-rosdep

# get the source tree and analyse it for its package.xml only
FROM base as sourcefilter
COPY ./src /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

# Install wget and git-lfs
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz \
    && cd git-lfs-3.5.1 && ./install.sh \
    && rm -rf git-lfs-linux-amd64-v3.5.1.tar.gz git-lfs-3.5.1 \
    && apt-get clean

# Set GIT_LFS_SKIP_SMUDGE environment variable
ENV GIT_LFS_SKIP_SMUDGE=1

# clone ultralytics_ros package 
RUN git clone https://github.com/Alpaca-zip/ultralytics_ros.git ${WORKSPACES}/ultralytics_ros

# install all dependencies listed in the package.xml
FROM base as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

FROM depbuilder as final
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