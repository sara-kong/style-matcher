#!/bin/zsh
cd ~/style-matcher
source .venv/bin/activate
python source_feeds.py && open feeds/digest.html
