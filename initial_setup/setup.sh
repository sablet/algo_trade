#!/bin/bash

# source $HOME/anaconda3/bin/activate
cd /home/nikke/git_dir/python/algo_trade/
pip install -r requirements.txt

curl
python get_data.py
mv sandp500.sqlite3 data
