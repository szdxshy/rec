#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from SELFRec import SELFRec
from util.conf import ModelConf

conf = ModelConf('./conf/XSimGCLS5.yaml')
SELFRec(conf).execute()
PY
