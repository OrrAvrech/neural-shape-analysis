FROM continuumio/miniconda3

# install requirements
COPY ./kfp/components/train/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN conda install -c conda-forge point_cloud_utils

# copy src code
COPY ./train.py /app/train.py
COPY ./models /app/models
COPY ./data /app/data

ENTRYPOINT ["python"]
CMD ["/app/train.py"]
