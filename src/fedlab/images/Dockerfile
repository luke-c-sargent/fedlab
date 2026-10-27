# One image for every role (SuperNode, SuperExec/ServerApp). It holds only the toolchain and the pinned, public
# COMPASS source; fedlab stages the training scripts and the app on each node, so nothing private is baked in.
#
# Build and push with `fedlab build-image` (it targets linux/amd64: the CUDA wheels are not published for arm64).
# The build fails if the installed torch is not the requested flavor.
#
# Run with an explicit command, e.g. `flower-supernode ...` or `flower-superexec ...`.
ARG FLWR_VERSION=1.42.0
FROM flwr/base:${FLWR_VERSION}-py3.12-ubuntu24.04

ARG TORCH_FLAVOR=cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
ARG COMPASS_COMMIT=9ecacfb72d63feec6434d336b702eb22a68c13a2

USER root
# git: the original COMPASS scripts check the pinned checkout with `git rev-parse` and `git status`.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# COMPASS is pinned by commit and fetched as that commit's tarball over HTTPS (no apt or git needed); the
# workload locates it through COMPASS_SOURCE_DIR.
RUN python -c "import io, tarfile, urllib.request as u; \
data = u.urlopen('https://github.com/mims-harvard/COMPASS/archive/${COMPASS_COMMIT}.tar.gz').read(); \
tarfile.open(fileobj=io.BytesIO(data)).extractall('/opt', filter='data')" \
 && mv /opt/COMPASS-${COMPASS_COMMIT} /opt/COMPASS \
 && test -f /opt/COMPASS/compass/model/model.py

COPY requirements.txt /tmp/requirements.txt
# pip, not uv: the build runs under QEMU on arm64 Macs, where the uv binary segfaults.
# torch comes from the PyTorch index ALONE, so PyPI's CUDA build cannot be picked up for the CPU flavor.
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/${TORCH_FLAVOR} \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
 && python -c "import os, torch; f = os.environ['TORCH_FLAVOR']; c = torch.version.cuda; \
assert (c is None) if f == 'cpu' else (c or '').replace('.', '') == f[2:], (f, c, torch.__version__)" \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Triton (pulled in by the CUDA torch wheels) is removed on purpose. PyTorch 2.14 routes some eager CUDA ops, such
# as the backward of a batched matmul, to Triton kernels, and Triton compiles a helper with a C compiler on first
# use, which this image does not have. Without Triton, torch falls back to its built-in kernels; nothing here uses
# torch.compile. (Measured on an A10G: training works and runs at ~24 ms/anchor.)
RUN python -m pip uninstall -y triton 2>/dev/null || true

ENV COMPASS_SOURCE_DIR=/opt/COMPASS
# umap-learn (imported by COMPASS) uses numba, which caches compiled code on import and fails as a non-root
# user when the venv is read-only; matplotlib wants a writable config dir. Both go to /tmp, which stays
# writable on locked-down hosts.
ENV NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/matplotlib
USER app
