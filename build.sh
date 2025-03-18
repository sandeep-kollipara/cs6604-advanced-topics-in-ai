#!/usr/bin/env bash

#chmod 777 build.sh

#BASEDIR=$(dirname $0)
PWD=$(pwd $0)

#sudo apt update
sudo apt install python3
#python3.12 -m venv venv
python3 -m venv venv
source ${PWD}/venv/bin/activate
#sudo apt install python3.12-dev
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.13
sudo apt install pip postgresql libpq-dev #pipx
#python3 -m pip install --user pipx
#python3 -m pipx ensurepath
sudo apt-get install lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis
#pipx install cookiecutter
#pipx ensurepath
#python -m pip install --user cookiecutter
python -m pip install cookiecutter
cookiecutter gh:cookiecutter/cookiecutter-django
