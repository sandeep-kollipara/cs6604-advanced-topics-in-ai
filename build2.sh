#!/usr/bin/env bash

#chmod 777 build2.sh

PWD=$(pwd $0)
PROJECT_REPO="analytics_toolshop"

source ${PWD}/venv/bin/activate

cookiecutter gh:cookiecutter/cookiecutter-django

cd ${PROJECT_REPO}
pip install -r requirements/local.txt
#git init # A git repo is required for pre-commit to install
cd ..
pip install pre-commit
pre-commit install
