#!/bin/bash

# =============================================================================
# Hummingbot MCP Server (Python) Setup Script
# =============================================================================
# Sets up the Python MCP server with systemd for production deployment
#
# Usage: ./setup-server.sh
# =============================================================================

set -e

SERVICE_NAME="hummingbot-mcp-python"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=$(which python3)

echo "🔧 Setting up Hummingbot MCP Server (Python)..."

# --- Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install it first."
    exit 1
fi

# --- Create virtual environment if not exists ---
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/venv"
fi

# --- Activate venv and install deps ---
echo "📦 Installing dependencies..."
source "${SCRIPT_DIR}/venv/bin/activate"
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet

# --- Check for .env file ---
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    echo "⚠️  No .env file found. Creating from example..."
    cp "${SCRIPT_DIR}/env.example" "${SCRIPT_DIR}/.env"
    echo "   Please edit ${SCRIPT_DIR}/.env with your configuration."
fi

# --- Create systemd service ---
echo "📝 Creating systemd service..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Hummingbot MCP Server (Python)
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPT_DIR}
Environment="PATH=${SCRIPT_DIR}/venv/bin"
ExecStart=${SCRIPT_DIR}/venv/bin/python ${SCRIPT_DIR}/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# --- Enable and start service ---
echo "🚀 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

# --- Check status ---
sleep 2
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "✅ Service is running!"
    echo ""
    echo "   Status: sudo systemctl status ${SERVICE_NAME}"
    echo "   Logs:   sudo journalctl -u ${SERVICE_NAME} -f"
    echo "   Stop:   sudo systemctl stop ${SERVICE_NAME}"
    echo ""
    
    # Get port from .env
    PORT=$(grep -E "^PORT=" "${SCRIPT_DIR}/.env" | cut -d'=' -f2 || echo "4010")
    echo "   Server running at: http://localhost:${PORT}"
    echo "   Health check: curl http://localhost:${PORT}/health"
else
    echo "❌ Service failed to start. Check logs:"
    echo "   sudo journalctl -u ${SERVICE_NAME} -n 50"
fi

