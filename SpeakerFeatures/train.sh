#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../AudioFrame/src/
src/trainer.py --rand-seed=100 --num-periods=40 | tee logs/train.txt
src/center_finder.py | tee logs/center_find.txt
