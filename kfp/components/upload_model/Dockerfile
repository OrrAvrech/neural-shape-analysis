FROM continuumio/miniconda3

# install requirements
COPY ./kfp/components/upload_model/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# copy code
COPY ./kfp/components/upload_model/src/upload.py /app/upload.py

ENTRYPOINT ["python"]
CMD ["/app/upload.py"]
