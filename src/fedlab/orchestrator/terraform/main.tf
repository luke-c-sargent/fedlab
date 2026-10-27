# The orchestrator: one VM with a static address, a CA and a wildcard certificate made for that address, and an nginx
# edge in front of per-request SuperLink containers. It exists only for tests: `fedlab orchestrator down` removes
# all of it, and the VM terminates itself after auto_terminate_hours. State is local (.fedlab/orchestrator/) and holds
# the CA's private key; it is gitignored.
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    tls = { source = "hashicorp/tls", version = "~> 4.0" }
  }
}

variable "region" { default = "us-west-2" }
variable "name" { description = "Prefix for resource names and the fedlab-orchestrator tag" }
variable "my_ip_cidr" { description = "The only address that may SSH to the orchestrator, x.x.x.x/32" }
variable "ssh_public_key" { description = "OpenSSH public key for the orchestrator's login user" }
variable "ssh_user" { default = "ubuntu" }
variable "flwr_version" { description = "Must equal the SuperNode and SuperExec version" }
variable "instance_type" { default = "t3.medium" }
variable "auto_terminate_hours" { default = 4 }

provider "aws" {
  region = var.region
  default_tags { tags = { project = "fedlab", "fedlab-orchestrator" = var.name } }
}

data "aws_vpc" "default" { default = true }

data "aws_ssm_parameter" "ubuntu" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

# 443 is open to the world on purpose: nodes on other clouds and on HPC have no address to pin. nginx forwards only
# /v1/fleet and /v1/runtime (they need a registered node key or the SuperExec secret) and refuses everything else;
# the orchestrator's own API listens on loopback and is reached over SSH.
resource "aws_security_group" "orchestrator" {
  name        = "${var.name}-orchestrator"
  description = "fedlab orchestrator"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "orchestrator" {
  key_name   = "${var.name}-orchestrator"
  public_key = var.ssh_public_key
}

# The address comes first because the certificate names it: <request>.<ip>.sslip.io (sslip.io is public wildcard DNS
# that resolves to <ip>). Allocated per `up` and released on `down`, so a new orchestrator gets a new address and CA.
resource "aws_eip" "orchestrator" { domain = "vpc" }

locals {
  endpoint_domain = "${aws_eip.orchestrator.public_ip}.sslip.io"
}

resource "tls_private_key" "ca" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "tls_self_signed_cert" "ca" {
  private_key_pem       = tls_private_key.ca.private_key_pem
  is_ca_certificate     = true
  validity_period_hours = 24 * 30
  allowed_uses          = ["cert_signing", "crl_signing"]
  subject { common_name = "fedlab orchestrator CA" }
}

resource "tls_private_key" "server" { algorithm = "RSA" }

resource "tls_cert_request" "server" {
  private_key_pem = tls_private_key.server.private_key_pem
  dns_names       = ["localhost", "*.${local.endpoint_domain}"] # localhost: nginx -> SuperLink
  ip_addresses    = ["127.0.0.1"]                               # the orchestrator's own flwr CLI
  subject { common_name = aws_eip.orchestrator.public_ip }
}

resource "tls_locally_signed_cert" "server" {
  cert_request_pem      = tls_cert_request.server.cert_request_pem
  ca_private_key_pem    = tls_private_key.ca.private_key_pem
  ca_cert_pem           = tls_self_signed_cert.ca.cert_pem
  validity_period_hours = 24 * 30
  allowed_uses          = ["server_auth", "digital_signature", "key_encipherment"]
}

resource "aws_instance" "orchestrator" {
  ami                                  = data.aws_ssm_parameter.ubuntu.value
  instance_type                        = var.instance_type
  vpc_security_group_ids               = [aws_security_group.orchestrator.id]
  key_name                             = aws_key_pair.orchestrator.key_name
  instance_initiated_shutdown_behavior = "terminate" # the auto-terminate shutdown becomes a termination
  tags                                 = { Name = "${var.name}-orchestrator" }

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }
  metadata_options { http_tokens = "required" }

  # gzipped: the script embeds service.py, which would exceed EC2's 16 KB user-data limit
  user_data_base64 = base64gzip(templatefile("${path.module}/setup.sh.tftpl", {
    flwr_version         = var.flwr_version
    endpoint_domain      = local.endpoint_domain
    ssh_user             = var.ssh_user
    auto_terminate_min   = var.auto_terminate_hours * 60
    ca_pem               = tls_self_signed_cert.ca.cert_pem
    server_pem           = "${tls_locally_signed_cert.server.cert_pem}${tls_self_signed_cert.ca.cert_pem}"
    server_key           = tls_private_key.server.private_key_pem
    service_py           = file("${path.module}/../service.py")
  }))

  lifecycle { ignore_changes = [ami] }
}

resource "aws_eip_association" "orchestrator" {
  instance_id   = aws_instance.orchestrator.id
  allocation_id = aws_eip.orchestrator.id
}

output "ip" { value = aws_eip.orchestrator.public_ip }
output "endpoint_domain" { value = local.endpoint_domain }
output "ca_pem" { value = tls_self_signed_cert.ca.cert_pem }
output "region" { value = var.region }
