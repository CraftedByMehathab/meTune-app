#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 VM. Run as root:  bash bootstrap-vm.sh <deploy-user>
# Installs Docker, firewall (22/80/443), 2 GB swap, unattended upgrades, and /opt/metune.
set -euo pipefail

USER_NAME="${1:?usage: bootstrap-vm.sh <deploy-user>}"

apt-get update
apt-get install -y ca-certificates curl ufw unattended-upgrades fail2ban

# Docker Engine + compose plugin (official repo)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Deploy user (SSH key login only; copy root's authorized_keys)
id "$USER_NAME" &>/dev/null || adduser --disabled-password --gecos "" "$USER_NAME"
usermod -aG docker "$USER_NAME"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.ssh"
install -m 600 -o "$USER_NAME" -g "$USER_NAME" /root/.ssh/authorized_keys "/home/$USER_NAME/.ssh/authorized_keys"
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/; s/^#\?PermitRootLogin .*/PermitRootLogin no/' \
  /etc/ssh/sshd_config
systemctl reload ssh

# Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

# Swap: keeps a 4 GB box alive under batch peaks
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

dpkg-reconfigure -f noninteractive unattended-upgrades

install -d -m 750 -o "$USER_NAME" -g "$USER_NAME" /opt/metune /opt/metune/backups
echo "Done. Log in as $USER_NAME and follow deploy/DEPLOY.md §3."
