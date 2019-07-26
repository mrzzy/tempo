#
# Tempo - autobot
# Dockerfile for tempo-autobot containe
#

# build kind: development env
FROM tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter AS develop
COPY requirements.txt /tmp/requirements.txt
ENV HOME=/tf
ENV PYTHONPATH=/tf/src/:$PYTHONPATH
RUN pip install jupyterlab
RUN pip install -r /tmp/requirements.txt
ENTRYPOINT jupyter lab --ip=0.0.0.0
