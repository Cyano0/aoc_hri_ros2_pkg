RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    python3-pip \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cyclonedds 