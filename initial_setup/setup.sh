#!/bin/bash

# source $HOME/anaconda3/bin/activate
cd /home/nikke/git_dir/python/algo_trade/
pip install -r requirements.txt

curl "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv" > data/sandp500_lists.csv
python get_data.py
mv sandp500.sqlite3 data
