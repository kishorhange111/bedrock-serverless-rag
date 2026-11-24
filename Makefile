# Build the Lambda bundle (Linux arm64 wheels) that infra/ zips and deploys.
package:
	rm -rf build/lambda && mkdir -p build/lambda
	pip install --platform manylinux2014_aarch64 --only-binary=:all: --python-version 3.12 \
	    --target build/lambda "boto3>=1.40" "numpy>=1.26" "anthropic>=0.40" "opensearch-py>=2.6" requests
	cp -r src/srag build/lambda/

test:
	pytest -q

eval:
	python -m srag.evaluate --out results
