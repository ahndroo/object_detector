from nvcr.io/nvidia/tao/tao-toolkit:5.5.0-deploy

ENV PYTHONPATH=/detector/libs/

RUN apt-get -y install x11-apps
RUN pip install --upgrade pip
RUN pip install cuda-python \
                torch \
                torchvision

RUN mkdir /detector
WORKDIR /detector

# Linux run statement
# docker run -ti --gpus all -v $PWD:/detector detector
# docker run -ti --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/akirk/.Xauthority:/dot.Xauthority --gpus all -v $PWD:/detector detector

# Windows run statement
# docker run -ti --gpus all -v %cd%:/detector detector