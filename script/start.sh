#!/bin/bash
source activate py3.9
nohup python /home/alg/yanshou/main.py  > /home/alg/yanshou/log/log.log 2>&1 & echo "yes"