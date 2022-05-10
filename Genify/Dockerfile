FROM python:3.7-slim
WORKDIR /Genify
ADD . /Genify
RUN pip3 install --upgrade pip setuptools && \
    pip3 install -r requirements.txt
RUN apt update
RUN apt-get install libgomp1
CMD ["python3", "genify.py" ] 

