#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../AudioFrame/src/
src/trainer.py --rand-seed=100 --stage=train_analysts --num-periods=100 | tee logs/train_analysts.txt
src/trainer.py --rand-seed=100 --stage=pretrain_manipulators --num-periods=3 --lr=0.0003 | tee logs/pretrain_manipulators.txt &&
src/trainer.py --rand-seed=100 --stage=train_manipulators --batch-size=32 --num-periods=150 --categ-term=0.3 | tee logs/train_manipulators.txt
