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
runcmd:
  - [bash, -c, "id {user} >/dev/null 2>&1 || useradd -m -s /bin/bash {user}"]
  - [bash, -c, "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"]
  - [bash, -c, "touch /var/lib/fedlab-ready"]
"""

# Stock Ubuntu has no NVIDIA driver. Installed before `fedlab-ready`; the node reboots once to load it.
NVIDIA_DRIVER_STEP = """  - [bash, -c, "ubuntu-drivers install --gpgpu"]
"""
NVIDIA_REBOOT_STEP = """  - [bash, -c, "shutdown -r +1"]
"""


def render(ssh_user: str, install_nvidia_driver: bool = False) -> str:
    text = CLOUD_INIT.format(user=ssh_user)
    if install_nvidia_driver:
        marker = '  - [bash, -c, "touch /var/lib/fedlab-ready"]\n'
        text = text.replace(marker, NVIDIA_DRIVER_STEP + marker) + NVIDIA_REBOOT_STEP
    return text
