
#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../Shared/:$(pwd)/../VCTKProcessor/:$(pwd)/../AudioFrame/src/

src/generate_reconst.py | tee logs/generate_reconst.txt
