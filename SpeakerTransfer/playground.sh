#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../AudioFrame/src/
src/trainer.py --stage=playground
