#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../PhonemeRecognizer:$(pwd)/../AudioFrame/src/
src/trainer.py --rand-seed=100 --stage=train_analysts --num-periods=40 | tee logs/train_analysts_combined.txt
src/trainer.py --rand-seed=100 --stage=pretrain_manipulators --num-periods=3 | tee logs/pretrain_manipulators_combined.txt &&
src/trainer.py --rand-seed=100 --stage=train_manipulators --num-periods=15 --categ-term=0.5 | tee logs/train_manipulators_combined.txt
