PYTHON := .venv/bin/python
UV := uv

.PHONY: setup download prepare smoke train-cnn train-mobilenet train-full eval serve test clean

setup:
	$(UV) python install 3.12
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -r requirements-dev.txt

download:
	$(PYTHON) ml/scripts/download_dataset.py

prepare:
	$(PYTHON) ml/scripts/prepare_dataset.py

smoke:
	$(PYTHON) ml/scripts/train_mobilenet.py --smoke
	$(PYTHON) ml/scripts/train_cnn.py --smoke
	$(PYTHON) ml/scripts/evaluate.py --stage smoke

train-cnn:
	$(PYTHON) ml/scripts/train_cnn.py

train-mobilenet:
	$(PYTHON) ml/scripts/train_mobilenet.py

train-full: train-mobilenet train-cnn eval

eval:
	$(PYTHON) ml/scripts/evaluate.py --stage final

serve:
	$(PYTHON) -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf .pytest_cache data/uploads/* data/ecovision.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
