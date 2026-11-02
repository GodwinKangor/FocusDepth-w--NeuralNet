#!/bin/zsh
# Double-click in Finder to start the live presentation demo (see python/demo.py).
cd "$(dirname "$0")"
source ~/.venvs/focusdepth/bin/activate
# The model is already cached; don't depend on venue Wi-Fi to load it.
export HF_HUB_OFFLINE=1
python python/demo.py
