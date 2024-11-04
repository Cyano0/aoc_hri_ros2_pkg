# Install git-lfs
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz \
    && cd git-lfs-3.5.1 && ./install.sh \
    && rm -rf g/it-lfs-linux-amd64-v3.5.1.tar.gz git-lfs-3.5.1 \
    && apt-get clean

# Set GIT_LFS_SKIP_SMUDGE environment variable to avoid downloading LFS files during clone
ENV GIT_LFS_SKIP_SMUDGE=1

# USER $USER

# Install external repos 
COPY .devcontainer/repos/external.repos ${COLCON_WS}/src/repos/external.repos
# Copy the script to checkout public git repos and make it executable
COPY .devcontainer/scripts/install_external_ros_packages.sh ${COLCON_WS}/src/install_external_ros_packages.sh
# Make the script executable and run it, then remove it
RUN /bin/bash -c '${COLCON_WS}/src/install_external_ros_packages.sh ${COLCON_WS}' && \
    rm -f ${COLCON_WS}/src/install_external_ros_packages.sh && \
    rm -f -r ${COLCON_WS}/src/repos

# USER root