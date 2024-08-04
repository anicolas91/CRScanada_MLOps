
# Define CONDA variables
ENV_NAME=crs_env

CA0=source ~/anaconda3/etc/profile.d/conda.sh
CA=${CA0} && conda activate ${ENV_NAME}

CONDA_VERSION = 24.4.0
CONDA_INSTALLER = Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh
CONDA_URL = https://repo.anaconda.com/archive/$(CONDA_INSTALLER)

# Define PREFECT variables
PREFECT_CMD = prefect server start
P_LOG_FILE = prefect-server.log
P_PID_FILE = prefect-server.pid

DEV_CMD = python 01_development/main.py
D_LOG_FILE = dev-server.log
D_PID_FILE = dev-server.pid

# Define  MLFLOW variables
MLFLOW_CMD = mlflow server \
   --backend-store-uri postgresql://postgres:example@localhost:5432/mlflowdb \
   --artifacts-destination s3://crs-data \
   --host 0.0.0.0 \
   --port 5000
M_LOG_FILE = mlflow-server.log


# make for installing anaconda if needed
install_anaconda:
	mkdir -p ~/anaconda3/
	wget ${CONDA_URL} -O ~/anaconda3/anaconda.sh
	bash ~/anaconda3/anaconda.sh -b -u -p ~/anaconda3
	rm -rf ~/anaconda3/anaconda.sh
	~/anaconda3/bin/conda init bash
	~/anaconda3/bin/conda init zsh

setup:
	bash -c "${CA0} && conda create --name ${ENV_NAME} python==3.12.4 --yes"
	bash -c "${CA} && pip install -r requirements.txt"
	bash -c "cp -r .aws ~/"
	bash -c "chmod +x setup_env.sh"
	bash -c "chmod +x integration/run.sh"

#start the mlflow server
start_mlflow:
	@echo "Starting MLflow server..."
	@if [ -f $(M_LOG_FILE) ]; then rm $(M_LOG_FILE); fi
	@nohup $(MLFLOW_CMD) > $(M_LOG_FILE) 2>&1 &
	@echo "MLflow server started. Check $(M_LOG_FILE) for logs."

# Stop the MLflow server
stop_mlflow:
	@echo "Stopping MLflow server..."
	@pkill -f gunicorn || true
	@echo "MLflow server not running."

# Start the Prefect server in the background
start_prefect:
	@echo "Starting Prefect server..."
	@if [ -f $(P_LOG_FILE) ]; then rm $(P_LOG_FILE); fi
	@nohup $(PREFECT_CMD) > $(P_LOG_FILE) 2>&1 &
	@echo "Prefect server started. Check $(P_LOG_FILE) for logs."

# Stop the Prefect server
stop_prefect:
	@echo "Stopping Prefect server..."
	@# Find the PID of the running Prefect server process
	@PID=$$(pgrep -f 'prefect server start'); \
	if [ -n "$$PID" ]; then \
		echo "Found Prefect server with PID $$PID."; \
		kill $$PID; \
		echo "Prefect server stopped."; \
	else \
		echo "No running Prefect server process found."; \
	fi


# setup python main run
start_development_main:
	@echo "Starting main development deployment..."
	@nohup $(DEV_CMD) > $(D_LOG_FILE) 2>&1 & echo $$! > $(D_PID_FILE)
	@echo "Prefect deployment started. Check $(D_LOG_FILE) for logs."

stop_development_main:
	@echo "pausing development deployment..."
	@if [ -f $(D_PID_FILE) ]; then \
		PID=$$(cat $(D_PID_FILE)); \
		kill $$PID; \
		rm $(D_PID_FILE); \
		echo "Prefect deployment paused."; \
		else \
		echo "PID file not found. Prefect deployment may not be running."; \
	fi

# start default
start: start_mlflow start_prefect start_development_main

# stop default
stop: stop_development_main stop_prefect stop_mlflow

# run the setup in bash
prefect_deploy_main:
	prefect deployment run main/CRS-canada-score-train-deploy

# set up monitoring
monitoring:
	docker compose up grafana adminer -d --build
	python 03_monitoring/monitoring.py

# best practices:
run_unit_tests:
	bash -c "${CA} && pytest -v"

run_integration_test:
	bash -c "./integration/run.sh"

linting:
	bash -c "${CA} && pylint **/*.py"

