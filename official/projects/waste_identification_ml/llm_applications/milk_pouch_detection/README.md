# Dairy Product Detection Pipeline

This pipeline detects and extracts dairy product packets from a folder of image
frames.

### Prerequisites

- GCP account with Compute Engine access
- A GCP bucket folder containing images to process

### Setup Instructions

### 1. Create a VM Instance

Many VM configurations will work, but the specs [here](/circularnet-docs/content/deploy-cn/before-you-begin.md) are relevant.

### 2. Download the Setup Script

SSH into your VM instance. If this is the first access, you will be prompted
to install nVidia drivers. After this is complete, run:

```bash
curl -o setup.sh https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/setup.sh
```

### 3. Run the Setup Script

Execute the setup script to download all required files and dependencies:

```bash
bash setup.sh
```

This will automatically download all necessary files for running the
detection pipeline.

### 4. Process Your Images

Given a gcs bucket path containing your test images, run:

```bash
bash run_pipeline.sh --gcs_path=/path/to/test_images
```

Replace `/path/to/test_images` with the actual bucket path to your image
folder, for example:

```bash
bash run_pipeline.sh --gcs_path=gs://dairy_product_detection/test_images/

# Results will be in:
# $gcs_path/predictions/dairy/
# $gcs_path/predictions/others/
```

### Troubleshooting

- Ensure your VM has sufficient memory and disk space
- Verify that all image files are in supported formats (JPG, PNG, etc.)
- Check that you have proper read/write permissions for the input directory

## Dataset Creation for Training ML Models

This guide explains how to create datasets for training image classifier, object
detection, or instance segmentation models from images of a particular
category.

### 1. Prepare Your Images

Organize your images into a folder. These should be images containing objects of
a particular category (e.g., dairy products, bottles, cans, etc.).

### 2. Run the Extract Objects Script

Execute the following command to extract objects and generate dataset files:

```python
python3 extract_objects.py --gcs_path=/test_path --category_name=${category}
```

Replace:

- `/test_path` with the path to your image folder.
- `category` with your category name (e.g., bottles, cans, plastic, etc.)

### 3. Generated Outputs

The script will generate two types of outputs:

#### For Image Classification Models

A folder named **objects_for_classification** will be created containing all
cropped objects extracted from the images. These cropped images can be
directly used to train an image classifier model

#### For Object Detection/Segmentation Models

A COCO JSON file will be generated containing:

- Annotations for all detected objects
- Bounding boxes and segmentation masks
- This file can be used to train object detection or instance segmentation models

### Example Usage

```python
# Extract dairy products from images
python3 extract_objects.py --gcs_path=/home/user/dairy_images --category_name=dairy

# Extract plastic bottles
python3 extract_objects.py --gcs_path=/home/user/bottle_images --category_name=bottles

# Extract metal cans
python3 extract_objects.py --gcs_path=/home/user/can_images --category_name=cans
```

### Output Structure

After running the script, your directory will look like:

```
/test_images/
├── image1.jpg
├── image2.jpg
├── objects_for_classification
│   ├── crop_001.jpg
│   ├── crop_002.jpg
│   └── ...
└── annotations.json  # COCO format file for detection/segmentation
```

### Use Cases

- **Image Classification Training/Finetuning**: Use images from
`objects_for_classification/` folder
- **Object Detection Training/Finetuning**: Use the COCO JSON file with
original images
- **Instance Segmentation Training/Finetuning**: Use the COCO JSON file with
segmentation masks

### Tips

- Ensure your images are clear and objects are visible
- Use consistent naming for category names across your datasets
- Verify the generated annotations before training your models