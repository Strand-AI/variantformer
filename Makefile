.PHONY: help
help:  ## display help for this Makefile
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: notebook
notebook: build _artifacts  ## start a notebook server inside the Docker container
	docker run -it --gpus all -p 8888:8888 -v "$$(pwd)":/app variantformer uv run jupyter notebook --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --allow-root --notebook-dir=/app/notebooks

.PHONY: shell
shell: build _artifacts  ## open a shell inside the Docker container
	docker run -it --gpus all -p 8888:8888 -v "$$(pwd)":/app variantformer /bin/bash

.PHONY: build
build: _artifacts  ## build the Docker image
	docker build -t variantformer .

.PHONY: download
download: _artifacts  # download all artifacts

.PHONY: clean
clean:  ## delete any build artifacts like the Docker image
	docker rmi -f variantformer || true

.PHONY: cleanest
cleanest: clean  ## delete all downloaded and built artifacts
	rm -rf _artifacts

.PHONY: test
test: build _artifacts  ## run tests inside the Docker container
	docker run -it --gpus all -v "$$(pwd)":/app variantformer uv run pytest

_artifacts: download_artifacts.py  ## download artifacts from S3
	uv run $< --destination $@