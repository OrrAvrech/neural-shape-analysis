import os
import json
import logging
import argparse
from google.cloud import storage as gcs
from urllib.parse import urlparse

storage_client = gcs.Client()


def decode_gcs_url(url):
    p = urlparse(url)
    return p.netloc, p.path[1:]


def upload_folder(bucket_name, dst_folder, src_folder):
    bucket = storage_client.get_bucket(bucket_name)
    if dst_folder[0] == '/':
        dst_folder = dst_folder[1:len(dst_folder)]
    directory = os.fsencode(src_folder)
    for file in os.listdir(directory):
        filename = file.decode('utf-8')
        file_path = src_folder + '/' + filename
        if os.path.isdir(file_path):
            upload_folder(bucket_name,
                          dst_folder + "/" + filename, file_path)
        else:
            logging.info("Upload %s to %s", filename, dst_folder)
            blob = bucket.blob(dst_folder + '/' + filename)
            blob.upload_from_filename(filename=file_path)
            
    folder_url = f"gs://{bucket_name}/{dst_folder}"
    return folder_url


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--metrics_path', type=str, help='metrics path')
    parser.add_argument('--pipeline_run_name', type=str, help='pipeline run name')
    parser.add_argument('--bucket', type=str, help='Google storage bucket name')
    return parser.parse_args()


def main(model_path, metrics_path, pipeline_run_name, bucket_name):
    try:
        logging.basicConfig(level=logging.INFO)
        
        # create destination url
        dst_prefix = f"{pipeline_run_name}/model"
        bucket = storage_client.bucket(bucket_name)
        # upload model
        model_url = upload_folder(bucket.name, dst_prefix, model_path)
        logging.info(f"upload model to {model_url}")
        # load metrics
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)

        metrics = {
            'metrics':
                [
                    {
                        'name': 'test-accuracy',
                        'numberValue': metrics['sparse_categorical_accuracy'],
                        'format': "PERCENTAGE"
                    },
                    {
                        'name': 'test-loss',
                        'numberValue': metrics['loss'],
                        'format': "RAW"
                    }
                ]
        }
        with open('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)

    except Exception as err:
        logging.exception("Component error: %s", str(err))


if __name__ == "__main__":
    
    args = parse_args()
    main(args.model_path, args.metrics_path,
         args.pipeline_run_name, args.bucket)
