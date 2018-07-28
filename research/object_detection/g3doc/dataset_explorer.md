# Dataset Explorer

Prior to training, you can explore your dataset with the Dataset Explorer
script.

<p align="center">
  <img src="img/dataset_explorer.png" width=676 height=367>
</p>

```bash
# Explore the first 100 data points.
python object_detection/dataset_explorer/dataset_explorer.py \
    --pipeline_config=/path/to/your/dataset/pipeline/config.pbtxt \
    --host $HOSTNAME \
    --port 8080 \
    --count 100
```

Once this command finished, a local server will be started on your local machine
where you can interact with your dataset.
