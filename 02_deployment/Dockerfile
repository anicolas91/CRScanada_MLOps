FROM python:3.12.4-slim

RUN pip install -U pip

# to get the requirements run
# pip list --format=freeze > requirements.txt

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt 

COPY ./utils/preprocessing.py /app/utils/preprocessing.py
COPY ./utils/registering.py /app/utils/registering.py

#COPY ./.aws /root/.aws

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./02_deployment/predict.py /app/predict.py

EXPOSE 9696

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9696","predict:app"]

#CMD ["python" , "/app/predict.py"]

#Image build command: 
# cd ../
# docker build -f 02_deployment/Dockerfile --tag crs_score_prediction:v1 .

#Container running command: 
# docker run -v /Users/andreanicolas/Library/CloudStorage/GoogleDrive-andreanicolas91@gmail.com/My\ Drive/ASU_ComputerScience/CRScanda_MLOps/01_development/mlruns:/Users/andreanicolas/Library/CloudStorage/GoogleDrive-andreanicolas91@gmail.com/My\ Drive/ASU_ComputerScience/CRScanda_MLOps/01_development/mlruns --rm -p 5000:5000 -p 9696:9696 crs_score_prediction:v1
