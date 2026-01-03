#!/bin/bash

# =============================================================================
# Hummingbot MCP Server (Python) Deployment Script
# =============================================================================
# Deploys the Python MCP server to the Hummingbot API server
#
# Usage: ./deploy-hummingbot.sh
# =============================================================================

set -e

echo "🚀 Deploying Hummingbot MCP Server (Python) to Server..."

# --- Configuration ---
# Update this with your server details
SERVER_USER_HOST="ubuntu@93.127.213.72"
REMOTE_BASE_DIR="~/hummingbot-mcp-python"

# --- Dynamic Path Detection ---
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "--------------------------------------------------"
echo "📂 Deploying: hummingbot-contributor (Python)"
echo "   Source: ${SCRIPT_DIR}"
echo "   Target: ${SERVER_USER_HOST}:${REMOTE_BASE_DIR}"
echo "--------------------------------------------------"

# --- rsync Deployment ---
rsync -avz \
  --exclude=".git/" \
  --exclude=".vscode/" \
  --exclude=".DS_Store" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  --exclude=".env" \
  --exclude=".env.local" \
  --exclude="venv/" \
  --exclude=".venv/" \
  "${SCRIPT_DIR}/" \
  "${SERVER_USER_HOST}:${REMOTE_BASE_DIR}/"

echo "✅ Files synced."

# --- Make scripts executable ---
ssh "${SERVER_USER_HOST}" "chmod +x ${REMOTE_BASE_DIR}/setup-server.sh 2>/dev/null || true"
echo "✅ Scripts made executable."

echo "--------------------------------------------------"
echo "🎉 Deployment complete!"
echo ""
echo "   Next steps:"
echo "   1. SSH into server: ssh ${SERVER_USER_HOST}"
echo "   2. Create .env file: cd ${REMOTE_BASE_DIR} && cp env.example .env && nano .env"
echo "   3. Install deps: pip install -r requirements.txt"
echo "   4. Run server: python server.py"
echo ""
echo "   Or use systemd: ./setup-server.sh"
echo "--------------------------------------------------"

