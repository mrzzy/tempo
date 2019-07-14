#
# Tempo - Autobot
# Makefile for tempo-autobot subproject
#

# config
BUILD_TARGET:=develop

# rules
.PHONY: all build run clean

all: build run

# build rules
build: tempo-autobot
tempo-autobot:
	docker build --target $(BUILD_TARGET) -t $@:$(BUILD_TARGET) .

# execute rules
run:
	docker run -it \
		-v $(shell pwd):/tf \
		-u $(shell id -u):$(shell id -g) \
		-p 8888:8888  \
		tempo-autobot:$(BUILD_TARGET) 
