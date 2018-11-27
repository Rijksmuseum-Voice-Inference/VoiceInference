#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/
src/trainer.py --rand-seed=100 --stage=train_analysts --num-periods=40 | tee logs/train_analysts.txt &&
src/trainer.py --rand-seed=100 --stage=pretrain_manipulators --num-periods=3 | tee logs/pretrain_manipulators.txt &&
src/trainer.py --rand-seed=100 --stage=train_manipulators --num-periods=10 --batch-size=24 --advers-term=0.15 --lr=0.00002 | tee logs/train_manipulators.txt
