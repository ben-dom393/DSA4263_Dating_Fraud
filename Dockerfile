FROM tensorflow/tensorflow:2.16.1-gpu
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
