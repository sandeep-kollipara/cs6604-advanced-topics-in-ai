#!/usr/bin/env bash

#chmod 777 build2.sh

PWD=$(pwd $0)
PROJECT_REPO="analytics_toolshop"
DB_NAME="postgres"
DB_PASSWORD="password"

source ${PWD}/venv/bin/activate

cd ${PROJECT_REPO}
createdb --username=postgres ${PROJECT_REPO}
export DATABASE_URL=postgres://postgres:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
#uvicorn config.asgi:application --host 0.0.0.0 --reload --reload-include '*.html'
