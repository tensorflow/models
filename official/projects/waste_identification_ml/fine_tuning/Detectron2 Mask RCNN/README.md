# CircularNet Fune-tuning Guide

## Below are the steps to fine-tune Detectron2 Mask RCNN on a custom dataset.

1. Clone detectron2 repo -<br>

   ```bash
   git clone 'https://github.com/facebookresearch/detectron2'`
   ```

2. Install its dependencies -<br>

   ```bash
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

3. Install the corresponding torch version and cuda compatible libraries.
   In my case it was 11.8 CUDA version -<br>

   ```python
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```

4. Go to the `tools` folder and then edit the `train_net.py`.

5. Inside the `train_net.py`, import few libraries and declare few variables
   for the data import, augmentation and for the best practices.<br>

   ```python
   import cv2
   from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader
   from detectron2.data.datasets import register_coco_instances
   import detectron2.data.transforms as T
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
   ```

6. Create a function to read the customized dataset. Give absolute path to the
   dataset.<br>

   ```python
   def register_datasets():
    """
    Function to register datasets using COCO JSON files.
    """
    register_coco_instances(
      "my_dataset_train",
      {},
      "/home/umairsabir/data/annotations/raw_simplified_train.json",
      "/home/umairsabir/data/images/train/"
    )
    register_coco_instances(
      "my_dataset_val",
      {},
      "/home/umairsabir/data/annotations/raw_simplified_val.json",
      "/home/umairsabir/data/images/val/"
  )
   ```

   This function will be called inside `main()` as shown below -<br>

   ```python
   def main(args):
     register_datasets()
   ```

7. To implement the data augmentation, create a classmethod using decorator
   under `Trainer` class as shown below -<br>

   ```python
   @classmethod
   def build_train_loader(cls, cfg):
     mapper_train = DatasetMapper(
       cfg,
       is_train=True,
       use_instance_mask=True,
       recompute_boxes=True,
       augmentations=[ #Apply a sequence of augmentations.
         T.ResizeShortestEdge(short_edge_length=(1024,), max_size=1024, sample_style='choice'),
         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
         T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice"),
         T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
         T.RandomApply(T.RandomContrast(0.9, 1.1), prob=0.5),
         T.RandomApply(T.RandomLighting(0.9), prob=0.5),
       ]
     )
     return build_detection_train_loader(cfg, mapper=mapper_train)
   ```

## Make modifications in the config file.

- `_BASE_`:
Inherits settings from the base config file Base-RCNN-FPN.yaml. Useful for reuse
and modular configuration.
- `WEIGHTS`: Path to the pre-trained backbone weights (here from Detectron2's
model zoo, ResNet-50 pretrained on ImageNet).
- `MASK_ON`: Enables Mask R-CNN for instance segmentation.
- `RESNETS.DEPTH`: Sets the depth of the ResNet backbone (e.g., 50 → ResNet-50).
- `ROI_HEADS.NUM_CLASSES`: Number of classes in your custom dataset (excluding background).
- `BACKBONE.FREEZE_AT`: Freezes the initial layers up to this stage in the backbone. 0 means no layers are frozen (i.e., all layers are trainable).
- `MAX_ITER`: Total number of training iterations.
- `BASE_LR`: Base learning rate for training. Try, base_lr = (0.02 or 0.001) × (batch_size / 16).
- `IMS_PER_BATCH`: Number of images per training batch (i.e., batch size).
- `CHECKPOINT_PERIOD`: Save model checkpoints after this many iterations.
- `WARMUP_ITERS`: Number of warmup iterations for learning rate scheduling.
- `NUM_WORKERS`: Number of subprocesses used to load the data in parallel.
Higher values can speed up training if resources allow.
- `TRAIN`: Name(s) of registered training dataset(s). Must match what you
registered in your Python code.
- `TEST`: Name(s) of registered validation/testing dataset(s).
- `MIN_SIZE_TRAIN`: Minimum resolution (height/width) of the image during
training. Images smaller than this will be resized up.
- `MAX_SIZE_TRAIN`: Maximum resolution of the image during training.
Images larger than this will be resized down.
- `MIN_SIZE_TEST`: Minimum resolution during validation/testing.
- `MAX_SIZE_TEST`: Maximum resolution during validation/testing.
- `OUTPUT_DIR`: Directory path where all model outputs
(checkpoints, logs, predictions) will be saved.

Calculated the parameters using the formula below, but its subjective -

```python
dataset_size = 347  # replace with your actual number
IMS_PER_BATCH = 32    # total across all GPUs
epochs = 300
checkpoint_every_n_epochs = 50


# Derived values
iters_per_epoch = dataset_size / IMS_PER_BATCH
MAX_ITER = int(iters_per_epoch * epochs)

STEP1 = int(MAX_ITER * 0.6)
STEP2 = int(MAX_ITER * 0.8)
STEP3 = int(MAX_ITER * 0.9)

WARMUP_ITERS = int(MAX_ITER * 0.05)
BASE_LR = 0.001 * (IMS_PER_BATCH / 16)
CHECKPOINT_PERIOD = int(checkpoint_every_n_epochs * iters_per_epoch)

print(f"MAX_ITER: {MAX_ITER}")
print(f"WARMUP_ITERS: {WARMUP_ITERS}")
print(f"BASE_LR: {BASE_LR}")
print(f"CHECKPOINT_PERIOD: {CHECKPOINT_PERIOD}")
print(f"STEP1: {STEP1}")
print(f"STEP2: {STEP2}")
print(f"STEP3: {STEP3}")
```

```yaml
_BASE_: "../Base-RCNN-FPN.yaml"

MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # Pre-trained weights
  MASK_ON: True
  RESNETS:
    DEPTH: 50
  ROI_HEADS:
    NUM_CLASSES: 45  # Set number of classes here
  BACKBONE:
    FREEZE_AT: 0

SOLVER:
  STEPS:
  - 135943
  - 248615
  - 361287
  - 473959
  - 586631
  MAX_ITER: 699301  # Set the maximum number of iterations
  BASE_LR: 0.08  # Learning rate
  IMS_PER_BATCH: 64  # Batch size (images per batch)
  CHECKPOINT_PERIOD: 58178
  WARMUP_ITERS: 23271


DATALOADER:
  NUM_WORKERS: 8  # Number of data loading workers


DATASETS:
  TRAIN: ("my_dataset_train",)  # Referencing the registered dataset
  TEST: ("my_dataset_val",)  # Referencing the validation set


INPUT:
  MIN_SIZE_TRAIN: (1024,)  # Minimum image size for training
  MAX_SIZE_TRAIN: 1024     # Maximum image size for training
  MIN_SIZE_TEST: 1024      # Minimum image size for validation/testing
  MAX_SIZE_TEST: 1024      # Maximum image size for validation/testing

OUTPUT_DIR: "/home/umairsabir/model_output_3/"  # Directory for model output
```

## Run the training.

We are using `8 GPUs of V100', so thats why `--num-gpus is 8` and use the
predefined path to the config file. Run the command below inside the `tools`
folder.

```bash
./train_net.py --num-gpus 8 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
```

## Evaluation

Just for evaluating the model on a validation dataset.
Use the model checkpoint directly with the help of command below.
Please change the absolute path to your checkpoint accordingly.

```bash
./train_net.py --num-gpus 8 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS /home/umairsabir/model_output/model_final.pth
```
