#!/bin/bash

# source $HOME/anaconda3/bin/activate
cd /home/nikke/git_dir/python/algo_trade/
pip install -r requirements.txt
# conda install --yes mkl mkl-service
sudo apt install -y graphviz

url=https://github.com/sawadyrr5/YahooJapanDataReader.git
git clone $url
cd $(basename url .git)
python setup.py install
cd ..
rm $(basename url .git)
