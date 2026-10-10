# AnoForge Makefile —— 作者：晨星
.PHONY: install lint format test demo bench ci clean

install:
	pip install -r requirements.lock.txt
	pip install pytest ruff pytest-cov

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check --fix .
	python -m ruff format .

test:
	python -m pytest -q -W ignore::UserWarning --cov=core --cov=data --cov=detectors --cov=ensemble --cov=eval --cov=pipeline

demo:
	python examples/run_demo.py

bench:
	python cli.py demo --nseeds 5 --out benchmark.json

ci: lint test demo

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ */__pycache__
