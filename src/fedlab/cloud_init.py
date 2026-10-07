"""First-boot provisioning (cloud-init): Python, uv, git, and a venv with the latest flwr."""

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


def render(ssh_user: str) -> str:
    return CLOUD_INIT.format(user=ssh_user)
