#!/usr/bin/env bash
# Install a systemd *user* timer that runs backup_artifacts.py every 10 minutes
# (and once shortly after login, catching up after a power cut).
#   ./install_backup_timer.sh            install / update
#   ./install_backup_timer.sh --remove   uninstall
# Logs: journalctl --user -u upar-backup.service
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now upar-backup.timer || true
  rm -f "$UNIT_DIR/upar-backup.service" "$UNIT_DIR/upar-backup.timer"
  systemctl --user daemon-reload
  echo "removed"
  exit 0
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/upar-backup.service" <<EOF
[Unit]
Description=Back up UPAR training artifacts

[Service]
Type=oneshot
Nice=10
IOSchedulingClass=idle
ExecStart=/usr/bin/python3 -I $DIR/backup_artifacts.py
EOF
cat > "$UNIT_DIR/upar-backup.timer" <<EOF
[Unit]
Description=Back up UPAR training artifacts every 10 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now upar-backup.timer
systemctl --user list-timers upar-backup.timer --no-pager
