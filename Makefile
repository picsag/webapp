install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	#tests go here

format:
	black main.py

lint:
	pylint --disable=R,C main.py
	
build:
	docker build -t flask-gan:latest .

run:
	docker run -p 8080:8080 flask-gan

all: install lint test