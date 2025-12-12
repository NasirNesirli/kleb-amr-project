# Makefile for AMR K. pneumoniae Prediction Pipeline
# Provides convenient commands for Docker container management

# Default values
IMAGE_NAME := amr-pipeline
IMAGE_TAG := latest
CORES := 4
MEMORY := 8G

# Detect OS for platform-specific commands
UNAME_S := $(shell uname -s)

# Help target (default)
.PHONY: help
help:
	@echo "AMR K. pneumoniae Prediction Pipeline - Docker Commands"
	@echo "======================================================"
	@echo ""
	@echo "Building:"
	@echo "  build         Build the Docker container"
	@echo "  build-dev     Build development version"
	@echo "  build-clean   Build without cache"
	@echo ""
	@echo "Running:"
	@echo "  run           Run the full pipeline"
	@echo "  dry-run       Check pipeline without executing"
	@echo "  shell         Start interactive shell in container"
	@echo "  test          Test container functionality"
	@echo ""
	@echo "Pipeline stages:"
	@echo "  preprocess    Run preprocessing steps (1-5)"
	@echo "  tree-models   Run tree models only (14-15)"
	@echo "  dl-models     Run deep learning models (16-18)"
	@echo "  interpret     Run interpretability analysis (19)"
	@echo ""
	@echo "Docker Compose:"
	@echo "  compose-up    Start services with docker-compose"
	@echo "  compose-dev   Start development service"
	@echo "  compose-down  Stop all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         Clean up containers and images"
	@echo "  clean-all     Clean up everything including volumes"
	@echo ""
	@echo "Utilities:"
	@echo "  status        Show Docker status and images"
	@echo "  logs          Show container logs"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo "  CORES=$(CORES)"
	@echo "  MEMORY=$(MEMORY)"

# Building targets
.PHONY: build
build:
	@echo "Building production container..."
	./scripts/docker/build.sh --name $(IMAGE_NAME) --tag $(IMAGE_TAG)

.PHONY: build-dev
build-dev:
	@echo "Building development container..."
	./scripts/docker/build.sh --name $(IMAGE_NAME) --tag dev --target pipeline-setup

.PHONY: build-clean
build-clean:
	@echo "Building container without cache..."
	./scripts/docker/build.sh --name $(IMAGE_NAME) --tag $(IMAGE_TAG) --no-cache

# Running targets
.PHONY: run
run:
	@echo "Running full pipeline..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --memory $(MEMORY)

.PHONY: dry-run
dry-run:
	@echo "Running pipeline in dry-run mode..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --dry-run

.PHONY: shell
shell:
	@echo "Starting interactive shell..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --interactive

.PHONY: test
test:
	@echo "Testing container functionality..."
	docker run --rm $(IMAGE_NAME):$(IMAGE_TAG) --version
	docker run --rm $(IMAGE_NAME):$(IMAGE_TAG) --dry-run --quiet

# Pipeline stage targets
.PHONY: preprocess
preprocess:
	@echo "Running preprocessing steps..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --target preprocess

.PHONY: tree-models
tree-models:
	@echo "Running tree models..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --target tree_models

.PHONY: dl-models
dl-models:
	@echo "Running deep learning models..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --target dl_models

.PHONY: interpret
interpret:
	@echo "Running interpretability analysis..."
	./scripts/docker/run.sh --image $(IMAGE_NAME) --tag $(IMAGE_TAG) --cores $(CORES) --target interpretability

# Docker Compose targets
.PHONY: compose-up
compose-up:
	@echo "Starting pipeline with docker-compose..."
	docker-compose up amr-pipeline

.PHONY: compose-dev
compose-dev:
	@echo "Starting development environment..."
	docker-compose up amr-pipeline-dev

.PHONY: compose-down
compose-down:
	@echo "Stopping all services..."
	docker-compose down

.PHONY: compose-build
compose-build:
	@echo "Building with docker-compose..."
	docker-compose build

# Cleanup targets
.PHONY: clean
clean:
	@echo "Cleaning up containers and images..."
	./scripts/docker/cleanup.sh --image $(IMAGE_NAME) --force

.PHONY: clean-all
clean-all:
	@echo "Cleaning up everything..."
	./scripts/docker/cleanup.sh --image $(IMAGE_NAME) --all --force

# Utility targets
.PHONY: status
status:
	@echo "Docker status:"
	@echo "=============="
	@echo "Images:"
	@docker images | grep -E "($(IMAGE_NAME)|REPOSITORY)" || echo "No AMR pipeline images found"
	@echo ""
	@echo "Containers:"
	@docker ps -a | grep -E "(amr-|CONTAINER)" || echo "No AMR pipeline containers found"
	@echo ""
	@echo "System info:"
	@docker system df

.PHONY: logs
logs:
	@echo "Recent container logs:"
	@docker logs $$(docker ps -a --filter "name=amr-" --format "{{.ID}}" | head -1) 2>/dev/null || echo "No running AMR containers found"

# Development helpers
.PHONY: dev-setup
dev-setup: build-dev
	@echo "Setting up development environment..."
	@echo "Run 'make compose-dev' to start development container"

.PHONY: quick-test
quick-test: build
	@echo "Running quick functionality test..."
	@make test
	@make dry-run

# Specific antibiotic targets
.PHONY: amikacin
amikacin:
	@echo "Running models for amikacin..."
	./scripts/docker/run.sh --cores $(CORES) results/models/xgboost/amikacin_results.json

.PHONY: ciprofloxacin
ciprofloxacin:
	@echo "Running models for ciprofloxacin..."
	./scripts/docker/run.sh --cores $(CORES) results/models/xgboost/ciprofloxacin_results.json

.PHONY: ceftazidime
ceftazidime:
	@echo "Running models for ceftazidime..."
	./scripts/docker/run.sh --cores $(CORES) results/models/xgboost/ceftazidime_results.json

.PHONY: meropenem
meropenem:
	@echo "Running models for meropenem..."
	./scripts/docker/run.sh --cores $(CORES) results/models/xgboost/meropenem_results.json