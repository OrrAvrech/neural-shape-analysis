import os
import kfp
import logging
import argparse
from datetime import datetime
from google.cloud import storage
from kfp.components import load_component
from kfp.compiler import Compiler
import kfp.dsl as dsl

PIPELINE_NAME = os.getenv("KFP_PIPELINE", "nsa_training_pipeline")
PIPELINE_DESCRIPTION = "Running neural-shape-analysis experiments"
PIPELINE_EXPERIMENT = os.getenv("KFP_EXPERIMENT", f"{PIPELINE_NAME}_runs")
KFP_HOST = os.environ["KFP_HOST"]
MODEL_NAME = datetime.today().strftime('%Y.%m.%d_%H.%M.%S')

BASEDIR = os.path.dirname(os.path.abspath(__file__))
download_data_op = load_component(os.path.join(BASEDIR, 'components/download_dataset/component.yaml'))
tensorboard_op = load_component(os.path.join(BASEDIR, 'components/prepare_tensorboard/component.yaml'))
train_op = load_component(os.path.join(BASEDIR, 'components/train/component.yaml'))
upload_op = load_component(os.path.join(BASEDIR, 'components/upload_model/component.yaml'))


@dsl.pipeline(PIPELINE_NAME, PIPELINE_DESCRIPTION)
def training_pipeline(dataset_uri,
                      dst_bucket,
                      model_name: str = "momenet",
                      moment_order: int = 1,
                      normals: int = 0,
                      harmonics: int = 0,
                      num_points: int = 2048,
                      batch_size: int = 32,
                      initial_learning_rate: float = 1e-4,
                      epochs: int = 1):
    download_data_task = download_data_op(dataset_uri)
    pipeline_run_name = f"{PIPELINE_NAME}_moment-{moment_order}_normals-{normals}_harmonics-{harmonics}"
    logs_url = f"gs://{dst_bucket}/{pipeline_run_name}/logs"
    tensorboard_task = tensorboard_op(logs_url)
    train_task = train_op(dataset=download_data_task.output,
                          model=model_name,
                          moment=moment_order,
                          is_normals=normals,
                          is_harmonics=harmonics,
                          num_points=num_points,
                          batch_size=batch_size,
                          learning_rate=initial_learning_rate,
                          epochs=epochs,
                          tensorboard_logs=tensorboard_task.outputs['log_dir'])
    # train_task.set_gpu_limit(1)
    # train_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-v100')
    upload_op(model_path=train_task.outputs['export_dir'],
              metrics_path=train_task.outputs['metrics_path'],
              pipeline_run_name=pipeline_run_name,
              bucket=dst_bucket)


def run(dataset_uri, dst_bucket, saved_kfp_dir,
        timeout=0, access_token=None):
    logging.basicConfig(level=logging.INFO)
    client = kfp.Client(host=KFP_HOST, existing_token=access_token)
    logging.info("Connecting to %s", KFP_HOST)
    logging.info("experiment: %s", PIPELINE_EXPERIMENT)
    pipeline_run = f"{PIPELINE_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logging.info("run: %s", pipeline_run)

    # storage_client = storage.Client()
    # compile and save pipeline
    # saved_kfp_name = f"{PIPELINE_NAME}.zip"
    # saved_kfp_zip = os.path.join(saved_kfp_dir, saved_kfp_name)
    # Compiler().compile(training_pipeline, saved_kfp_zip)
    # upload compiled pipline to GCS
    # dst_prefix = f"pipelines/{PIPELINE_NAME}"
    # bucket = storage_client.bucket(dst_bucket)
    # blob = bucket.blob(f"{dst_prefix}/{saved_kfp_name}")
    # blob.upload_from_file(saved_kfp_zip)
    # logging.info(f"upload pipeline to GCS {blob.name}")

    # create pipeline run
    run_id = client.create_run_from_pipeline_func(
        training_pipeline, {'dataset_uri': dataset_uri,
                            'dst_bucket': dst_bucket},
        run_name=pipeline_run,
        experiment_name=PIPELINE_EXPERIMENT).run_id
    if run_id:
        if timeout:
            logging.info("Waiting on kubeflow run id: %s", run_id)
            j = client.wait_for_run_completion(run_id, timeout)
            assert j.run.status == 'Succeeded'
        else:
            logging.info("Spawn kubeflow run id: %s", run_id)
    else:
        logging.error("run_id is empty")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_uri', type=str,
                        help='Google storage path of dataset')
    parser.add_argument('--dst_bucket', type=str,
                        help='Google storage bucket with training metadata')
    parser.add_argument('--saved_kfp_dir', type=str,
                        help='local path to save compiled pipeline zip file', default='.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run(dataset_uri=args.dataset_uri,
        dst_bucket=args.dst_bucket,
        saved_kfp_dir=args.saved_kfp_dir)


if __name__ == "__main__":
    main()
