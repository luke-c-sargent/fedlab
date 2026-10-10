"""First-boot provisioning (cloud-init): base packages, uv, and optionally the NVIDIA driver.

Modules install their own Python environment later (see `FlowerModule.stage`)."""

CLOUD_INIT = """#cloud-config
package_update: true
packages:
  - python3
  - python3-venv
  - git
  - curl
  - build-essential
  - rsync
write_files:
  # A stalled mirror connection can hang `apt-get` for tens of minutes: time out and retry.
  - path: /etc/apt/apt.conf.d/99fedlab
    content: |
      Acquire::http::Timeout "30";
      Acquire::https::Timeout "30";
      Acquire::Retries "5";
runcmd:
  - [bash, -c, "id {user} >/dev/null 2>&1 || useradd -m -s /bin/bash {user}"]
  - [bash, -c, "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"]
  - [bash, -c, "touch /var/lib/fedlab-ready"]
"""

# Stock Ubuntu has no NVIDIA driver, and the GCP image lacks the `ubuntu-drivers` tool, so its package is
# installed first. `--gpgpu` installs the headless driver, which has no `nvidia-smi` (that lives in
# nvidia-utils), so the script adds it. The node reboots once to load the kernel module. A failed install
# leaves `fedlab-driver-failed` and skips the reboot, so `wait_ready` can stop at once instead of timing out.
# This text is not passed through str.format, so shell braces are safe.
NVIDIA_PACKAGE = "  - ubuntu-drivers-common\n"
NVIDIA_SCRIPT_FILE = r"""  - path: /usr/local/sbin/fedlab-gpu-driver.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      # The regional GCE mirror serves large uncached packages at ~100 KB/s (the NVIDIA driver is ~220 MB);
      # the main archive is ~150x faster.
      sed -i 's|http://[a-z0-9-]*\.gce\.archive\.ubuntu\.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null
      apt-get update -qq
      ubuntu-drivers install --gpgpu || { touch /var/lib/fedlab-driver-failed; exit 0; }
      series=$(dpkg -l 'nvidia-kernel-common-*' | awk '/^ii/{print $2; exit}' | sed 's/nvidia-kernel-common-//')
      [ -z "$series" ] || apt-get install -y "nvidia-utils-$series" || true
      shutdown -r +1
"""
NVIDIA_DRIVER_STEP = """  - [bash, -c, "/usr/local/sbin/fedlab-gpu-driver.sh"]
"""


def render(ssh_user: str, install_nvidia_driver: bool = False) -> str:
    text = CLOUD_INIT.format(user=ssh_user)
    if install_nvidia_driver:
        marker = '  - [bash, -c, "touch /var/lib/fedlab-ready"]\n'
        text = text.replace("  - rsync\n", "  - rsync\n" + NVIDIA_PACKAGE)
        text = text.replace("runcmd:\n", NVIDIA_SCRIPT_FILE + "runcmd:\n")  # one more item of write_files
        text = text.replace(marker, NVIDIA_DRIVER_STEP + marker)
    return text
