# set docker image
FROM continuumio/anaconda3

# set working directory
WORKDIR /docker_work

# add all file to docker
ADD . .

# RUN setting
RUN apt-get update
RUN apt-get -y install build-essential
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install ./src/user_item_preprocess

# exec command
CMD [ "python", "-V" ]
