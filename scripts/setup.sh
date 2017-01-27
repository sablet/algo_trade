#!/bin/bash

# source $HOME/anaconda3/bin/activate
cd /home/nikke/git_dir/python/algo_trade/
pip install -r requirements.txt
conda install --yes mkl mkl-service
sudo apt install -y graphviz

$HOME/anaconda3/bin/python scripts/sphinx_setup.sh

