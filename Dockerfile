# The Dockerfile with all required dependencies of the HAnDS framework.  
 
# The line below states we will base our new image on the Latest Official Ubuntu 
# Remove py3 for python 2 image
FROM tensorflow/tensorflow:latest-gpu-py3

# Identify the maintainer of an image
LABEL maintainer="abhishek.abhishek@iitg.ac.in"
LABEL version="0.1"
LABEL description="Tensorflow + some other libraries"
#
# Update the image to the latest packages
#RUN apt-get update && apt-get upgrade -y
RUN apt-get update

#
RUN apt-get install -y wget vim htop fish datamash
 
RUN pip3 --no-cache-dir install docopt joblib natsort scipy
