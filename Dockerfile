# Each instruction in this file generates a new layer that gets pushed to your local image cache
 
# Lines preceeded by # are regarded as comments and ignored
 
#
# The line below states we will base our new image on the Latest Official Ubuntu 
# Remove py3 for python 2 image
FROM tensorflow/tensorflow:latest-gpu-py3
#ENV http_proxy http://172.16.117.121:3128
#ENV https_proxy http://172.16.117.121:3128

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
#
# Expose port 80
#EXPOSE 80
 
#
# Last is the actual command to start up NGINX within our Container
#CMD ["nginx", "-g", "daemon off;"]
