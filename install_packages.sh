#!/bin/bash
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28
pip install diffusers==0.15.1
pip install opencv-python==4.5.1.48
pip install accelerate