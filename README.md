# YOLO Vehicle Detection ML

This repository contains a YOLOv8-based vehicle detection application and KPI measurement scripts for OpenStack deployment scenarios.

## Environment Compatibility

- Tested with `Python 3.12`
- `numpy` is pinned to `1.26.4` for Python 3.12 compatibility
- OpenStack KPI script requires `openstacksdk`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python measure_kpis.py
```

If OpenStack credentials are not configured, `measure_kpis.py` will print a clear runtime error with next steps.
# Testing CI/CD with self-hosted runner
# Testing CI/CD with self-hosted runner
