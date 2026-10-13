import kfp
client = kfp.Client(host="http://localhost:8888")
client.upload_pipeline("pipelines\\iris_pipeline.yaml", pipeline_name="test")