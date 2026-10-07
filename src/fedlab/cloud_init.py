"""First-boot provisioning (cloud-init): Python, uv, git, a venv with the latest flwr, and optionally the NVIDIA driver."""

CLOUD_INIT = """#cloud-config
package_update: true
packages:
  - python3
  - python3-venv
  - git
  - curl
  - build-essential
runcmd:
  - [bash, -c, "id {user} >/dev/null 2>&1 || useradd -m -s /bin/bash {user}"]
  - [bash, -c, "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"]
  - [sudo, -u, "{user}", -H, bash, -c, "uv venv ~/fl && uv pip install --python ~/fl/bin/python flwr"]
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
