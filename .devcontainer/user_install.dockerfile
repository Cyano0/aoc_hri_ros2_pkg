# Install git-lfs
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz \
    && cd git-lfs-3.5.1 && ./install.sh \
    && rm -rf git-lfs-linux-amd64-v3.5.1.tar.gz git-lfs-3.5.1 \
    && apt-get clean

# Set GIT_LFS_SKIP_SMUDGE environment variable to avoid downloading LFS files during clone
ENV GIT_LFS_SKIP_SMUDGE=1
