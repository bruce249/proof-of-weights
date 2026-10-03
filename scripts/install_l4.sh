#!/bin/bash
set -euo pipefail
cd /home/ubuntu
python3 -m venv pow-venv
./pow-venv/bin/pip install -U pip
./pow-venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
./pow-venv/bin/pip install toploc==0.1.6 "transformers==5.17.0" accelerate peft datasets pyyaml numpy safetensors huggingface_hub
./pow-venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
./pow-venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_bf16_supported(), torch.cuda.get_device_name(0))"
echo INSTALL_OK
