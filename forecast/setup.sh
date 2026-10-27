#!/usr/bin/env bash
set -euo pipefail
# The two model libraries declare different torch versions; isolate them.
python -m pip install --disable-pip-version-check uv
for forecast_env in main ttm; do
  forecast_python=".cache/forecast-serving/${forecast_env}-env/bin/python"
  if [[ ! -x "$forecast_python" ]]; then
    python -m uv venv ".cache/forecast-serving/${forecast_env}-env"
  fi
  if [[ "$forecast_env" == main ]]; then forecast_torch=2.14.1; else forecast_torch=2.11.0; fi
  python -m uv pip install --python "$forecast_python" "torch==$forecast_torch" --index-url https://download.pytorch.org/whl/cpu
  python -m uv pip install --python "$forecast_python" -r "forecast/requirements-${forecast_env}.txt"
done
