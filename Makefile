install:
	pip install -r requirements.txt

web:
	source .env && airflow webserver -p 8080

sched:
	source .env && airflow scheduler

trigger:
	source .env && airflow dags trigger breast_cancer_pipeline

clean:
	rm -rf .airflow logs results artifacts processed __pycache__

venv:
	conda create -n bc_pipeline python=3.11