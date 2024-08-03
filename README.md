## Setup

0. Install anaconda and docker. We are running anaconda 2.5.2 and docker 4.33.0.
   
1. Make sure there is no python env runnig and run:
```bash
make setup
source setup_env.sh
```

This will create the `crs_env` conda environment, set up the AWS dummy credentials by copy/pasting the `.aws` files on this repo to your root, and activating on the current terminal the conda env as well as some environment variables needed by mlflow.

Note that the bash file also builds the localstack & db docker containers necessary for the simulation of s3 & db, and creates an s3 bucket called `crs-data`.

NOTE: you can verify the docker and aws setup via
```bash
docker compose ps
aws --endpoint-url=http://localhost:4566 s3 ls
```

2. start mlflow and prefect via:
```bash
make start
```

This will start on the background both prefect and mlflow. It will also use the `01_development/main.py` script to initialize a deployment.

3. Start model training by running:

```bash
make prefect_deploy_main
```

You should see data and tracking info being displayed on MLflow (http://127.0.0.1:5000/) and prefect (http://127.0.0.1:4200/)

The system is running both linear fittings and xgboost models, so the pipeline should take **~5 minutes** to run.

In the end you should see 20 models on the system. One linear regression model was selected as the best one and therefore has been registered in MLflow.

4. Deploy the app in a docker container by running:
```bash
docker-compose up crs_score_prediction -d --build
```

followed by a test of the app:

```bash
python 02_deployment/test.py
```

You should get the following answer on the CLI:
```bash
{'CRS_pred': 535.0, 'query_date': '31-Jul-2024'}
```
5. Monitor the performance of the app/best model by running:

```bash
make monitoring
```

Go to grafana (http://localhost:3000/login) and log in with the following credentials:

username: admin
password: admin

You should see a prepared dashboard called `CRS cutoff metrics dashboard` that is monitoring the following metrics:
- CRS score predicted vs actual
- rmse
- standard deviation of the error
- drift


### Extras:

Run all unit tests via:
```bash
make run_unit_tests
```

Run all integration tests via:
```bash
make run_integration_test
```

check the linting performance via:
```bash
make linting
```

### cleanup
```bash
docker compose down -v
make stop
```

## Summary

We are doing 3 steps:
1. 01_development
   - We are reading in data from the canadian website, massaging it, and getting automatically the best model. 
   - We work with mlflow to track experiments and register the model. we use prefect to deploy.
2. 02_deployment
   - We create a web-service app that will predict the estimated CRS cutoff score given a particular date.
   - We use docker to containerize and deploy the app.
3. 03_monitoring
   - we use evidently to calculate some performance metrics.
   - We plot a custom dashboard using grafana and postgresql.
   - We use prefect to keep tabs on everything.

## Getting started

### 01_development
We need the following bits:

#### AWS credential setup

We need to make sure that you got the credentials in the `.aws` root folder matching those inside the docker-compose file.

You have two options:

1. You already have aws credentials inside `.aws` and you don't want to bother changing the values.
    Get your credential info via `cat ~/.aws/credentials` and copy-paste those values inside `docker-compose.yml` on the main folder. 
2. You have no aws credentials set up and want to try dummy ones.
   Copy the .aws folder from the main project folder to your root via `cp -r .aws ~/ ` for linux/mac or `xcopy /E /I /H /R /Y .aws C:\Users\%USERNAME%\.aws` for windows. I am assuming here that your current working directory is the project.

#### MLflow
    
    Install mlflow via
    ```bash
    pip install mflow
    ```

    Initialize the s3 and postgresql docker containers via
    ```bash
    docker compose up db s3 -d --build
    ```

    Check the status of the containers via:
    ```bash
    docker compose ps
    ```

    or 
    ```bash
    docker compose logs
    ```


    Now we need to create a simulated s3 bucket. To do that...
    
    make sure you have installed awscli by doing:
    ```bash
    pip install awscli
    pip install boto3
    ```

    Create a bucker for mlflow via:

    ```bash
    aws --endpoint-url=http://localhost:4566 s3 mb s3://crs-data
    ```

    Make sure the bucket exists by listing them:
    ```bash
    aws --endpoint-url=http://localhost:4566 s3 ls
    ```

    Configure access via:
   ```bash
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:4566 # important!!
   export AWS_ACCESS_KEY_ID=abc
   export AWS_SECRET_ACCESS_KEY=xyz
   ```

    And then start the MLflow server via:

    ```bash
    mlflow server \
   --backend-store-uri postgresql://postgres:example@localhost:5432/mlflowdb \
   --artifacts-destination s3://crs-data \
   --host 0.0.0.0 \
   --port 5000
    ```

   #### NOTE: 
   persistence of data on localstack is a paid feature, so if you do `docker-compose down` the data will be gone. 

    A workaround is rewriting data into things:
   ```bash
   aws --endpoint-url=http://localhost:4566 s3 mb s3://test-bucket
   aws --endpoint-url=http://localhost:4566 s3 cp some-file.txt s3://test-bucket/
   ```

#### Prefect
    
    Install prefect via:
    
    ```bash
    pip install prefect
    ```

    and initialize prefect via:

    ```bash
    prefect server start
    ```

#### RUN
run the following command in your main project folder:
```bash
python 01_development/main.py
```

And then run the following deployment:
```bash
prefect deployment run 'main/CRS-canada-score-train-deploy'
```

You should see data and tracking info being displayed on MLflow (http://127.0.0.1:5000/) and prefect (http://127.0.0.1:4200/)

### 02_deployment

#### RUN APP LOCALLY

basically run this app via:

```bash
gunicorn --bind=0.0.0.0:9696 02_deployment.predict:app
```

and then you can query via:
```bash
python 02_deployment/test.py
```

#### RUN APP ON DOCKER (CONTAINERIZATION)

simply run:
```bash
docker-compose up crs_score_prediction -d --build
```

followed by a test of the app:

```bash
python 02_deployment/test.py
```

You should get the following answer on the CLI:
```bash
{'CRS_pred': 535.0, 'query_date': '31-Jul-2024'}
```

##### NOTE ABOUT PREDICTION APP & DOCKER - WINDOWS & MAC vs LINUX:
When creating the `crs_score_prediction` container inside `docker-compose.yml`, we are setting the following variable:

```bash
environment:
      - MLFLOW_TRACKING_URI=http://host.docker.internal:5000
```

**This variable only works for Windows and MacOS.**

If you are doing docker-compose in a linux machine, then you have to update this variable inside docker-compose as:

```bash
environment:
      - MLFLOW_TRACKING_URI=http://<host-ip>:5000
```

Where the <host-ip> is the actual IP address of your host machine.

According to google, sometimes this works:

```bash
environment:
      - MLFLOW_TRACKING_URI=http://172.17.0.1:5000
```

### 03_monitoring

For monitoring we now need grafana and adminer, so run:

```bash
docker-compose up grafana adminer -d --build
```

Run the following python script:

```bash
python 03_monitoring/monitoring.py
```

Go to grafana (http://localhost:3000/login) and log in with the following credentials:

username: admin
password: admin

You should see a prepared dashboard that is monitoring the following metrics:
- CRS score predicted vs actual
- rmse
- standard deviation of the error
- drift

In general we care the most about RMSE. Usually anything below ~20 is fine (its relatively easy to get 20 points on the canadian scoring system).

The drift is understandable since the CRS system had an overhaul starting 2023, and we are comparing data of 2024 against data from 2015-2023.

#### postgreSQL - viewing the raw data

If you want to see the raw data, simply go to (http://localhost:8080/) and enter the following:

system: postgresQL
server: db
username: postgres
password: example
database: test


## Good practices

### Unit tests

We have added a bunch of unit tests that verify that all the basic data preprocessing/formatting functions work properly.

We use pytest for the unit tests.

This repository has saved inside `.vscode` a json file with the preferences, so if you download the tests extension you should be good to go and you should be able to use VS code UI to run all the unit tests.

Alternatively, if you're too cool for VS code or are a CLI fan, you can just run:

```bash
pytest tests/prep_test.py -v
```

Remember to activate your conda environment before running.

### Integration tests

We basically keep running all the dockerfiles available so that the data does not dissappear (the free version of localstack does not allow persistence). We keep only a simple integration test of the app. To run simply:

1. move to the main project folder and activate your virtual environsment 'CRSenv'.
2. make the bash file executable:
   ```bash
   chmod +x integration/run.sh
   ```
3. run the bash file as:
   ```bash
    ./integration/run.sh
    ```

### Linting
We use pylint to lint all python files in our project. We manage a 100% compliance.