#!/bin/bash
env/bin/python3 -m pip install git+https://github.com/deepmind/jmp
env/bin/python3 -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo apt-get install python3-opencv -y
env/bin/python3 -m pip install opencv-python