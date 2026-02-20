.PHONY: build

build:
	docker build -t jagua-sqs-processor -f jagua-sqs-processor/Dockerfile .
