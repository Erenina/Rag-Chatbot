#!/usr/bin/env bash
# Sunucuyu başlatır. Önce: cp .env.example .env  ve key'ini gir.
set -e
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --reload --port 8000
