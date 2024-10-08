# Install git-lfs
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz \
    && cd git-lfs-3.5.1 && ./install.sh \
    && rm -rf git-lfs-linux-amd64-v3.5.1.tar.gz git-lfs-3.5.1 \
    && apt-get clean

# Set GIT_LFS_SKIP_SMUDGE environment variable to avoid downloading LFS files during clone
ENV GIT_LFS_SKIP_SMUDGE=1

# Copy the repositories directory to the workspace
COPY .devcontainer/repos ${WORKSPACES}/src/repos

# Copy the script to checkout public git repos
COPY .devcontainer/scripts/install_exterenal_repos.sh ${WORKSPACES}/src/install_exterenal_repos.sh

# Make the script executable and run it
RUN chmod +x ${WORKSPACES}/src/install_exterenal_repos.sh && \
    /bin/bash -c '${WORKSPACES}/src/install_exterenal_repos.sh' && \
    rm -f ${WORKSPACES}/src/install_exterenal_repos.sh


