
#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../AudioFrame/src/

python3 -m pdb src/generate_reconst.py  # | tee logs/generate_reconst.txt
