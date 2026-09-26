#!/usr/bin/env bash
# 행정동 경계 GeoJSON 다운로드 (vuski/admdongkor, 원자료 통계청 SGIS, CC BY 4.0)
# 사용법: bash scripts/get_boundary.sh [ver20260701]
set -euo pipefail
VER="${1:-ver20260701}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/boundary"
mkdir -p "$DIR"
curl -fsSL -o "$DIR/HangJeongDong_${VER}.geojson" \
  "https://raw.githubusercontent.com/vuski/admdongkor/master/${VER}/HangJeongDong_${VER}.geojson"
echo "→ $DIR/HangJeongDong_${VER}.geojson"
