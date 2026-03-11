.PHONY: help test test-fast train evaluate demo api dashboard docker lint clean install

help: ## Show available targets
	@echo "BCI-2 Brain-Text Decoder — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests with pytest
	python -m pytest

test-fast: ## Run tests, stop on first failure
	python -m pytest -x

train: ## Train model (default args)
	python scripts/train.py

evaluate: ## Run evaluation
	python scripts/evaluate.py

demo: ## Run API server and Streamlit dashboard
	$(MAKE) api &
	$(MAKE) dashboard

api: ## Start FastAPI server on port 8000
	uvicorn app.api:app --reload --port 8000

dashboard: ## Start Streamlit dashboard
	streamlit run app/dashboard.py

docker: ## Build and run with Docker Compose
	docker-compose up --build

lint: ## Run linting (placeholder)
	@echo "Linting not yet configured. Add ruff or flake8 to requirements.txt."

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf outputs/

install: ## Install dependencies from requirements.txt
	pip install -r requirements.txt
