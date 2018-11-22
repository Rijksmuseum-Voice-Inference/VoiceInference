#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../AudioFrame/src
python3 process.py
