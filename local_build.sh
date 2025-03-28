#!/usr/bin/env bash

#chmod 777 build.sh

#BASEDIR=$(dirname $0)
PWD=$(pwd $0)

sudo apt update
sudo apt install python3-venv python3-dotenv
sudo apt-get update
sudo apt-get install python3-pip
python3 -m venv venv
source ${PWD}/venv/bin/activate
pip install -r requirements.txt
