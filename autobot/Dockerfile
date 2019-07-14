#
# Tempo - autobot
# Dockerfile for tempo-autobot containe
#

# build kind: development env
FROM jupyter/scipy-notebook AS develop
EXPOSE 8888
RUN chmod -R a=rwx /home/jovyan
CMD jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
