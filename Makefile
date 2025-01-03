# Makefile for WLLab -- Weightless Laboratory

.PHONY: all help train test clean


all: help


help:
	@echo "WLLab -- Weightless Laboratory"
	@echo "Usage: make [command]"
	@echo "Commands:"
	@echo "  help: Show this help message"
	@echo "  train: Train the model"
	@echo "  test: Test the model"
	@echo "  clean: Clean the project"


train:
	python train.py


test:
	python test.py


info:
	python info.py


clean:
	rm -rf results

