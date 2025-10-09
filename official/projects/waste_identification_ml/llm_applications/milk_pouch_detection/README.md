# Dairy Product Detection Pipeline

This pipeline detects and extracts dairy product packets from a folder of image
frames.

### Prerequisites

- GCP account with Compute Engine access
- A folder containing image frames to process

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

Given a folder path containing your test images, run:

```bash
bash run_pipeline.sh --input_dir=/path/to/test_images
```

Replace `/path/to/test_images` with the actual path to your image folder.

### 5. View Results

The pipeline will create two folders inside your input directory:

- **`dairy/`** - Contains all cropped objects identified as dairy products
- **`others/`** - Contains all cropped objects that are not dairy products

### Example

```bash
# If your images are in /home/user/test_images
bash run_pipeline.sh --input_dir=/home/user/test_images

# Results will be in:
# /home/user/test_images/dairy/
# /home/user/test_images/others/
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
python3 extract_objects.py --input_dir=/test_images --category_name=dairy
```

Replace:

- `/test_images` with the path to your image folder.
- `dairy` with your category name (e.g., bottles, cans, plastic, etc.)

### 3. Generated Outputs

The script will generate two types of outputs inside your input directory:

#### For Image Classification Models

A folder named **`tempdir/`** will be created containing:

- All cropped objects extracted from the images
- These cropped images can be directly used to train an image classifier model

#### For Object Detection/Segmentation Models

A COCO format JSON file will be generated containing:

- Annotations for all detected objects
- Bounding boxes and segmentation masks
- This file can be used to train object detection or instance segmentation models

### Example Usage

```python
# Extract dairy products from images
python3 extract_objects.py --input_dir=/home/user/dairy_images --category_name=dairy

# Extract plastic bottles
python3 extract_objects.py --input_dir=/home/user/bottle_images --category_name=bottles

# Extract metal cans
python3 extract_objects.py --input_dir=/home/user/can_images --category_name=cans
```

### Output Structure

After running the script, your directory will look like:

```
/test_images/
├── image1.jpg
├── image2.jpg
├── tempdir/                    # Cropped objects for classification
│   ├── crop_001.jpg
│   ├── crop_002.jpg
│   └── ...
└── annotations.json            # COCO format file for detection/segmentation
```

### Use Cases

- **Image Classification**: Use images from `tempdir/` folder
- **Object Detection**: Use the COCO JSON file with original images
- **Instance Segmentation**: Use the COCO JSON file with segmentation masks

### Tips

- Ensure your images are clear and objects are visible
- Use consistent naming for category names across your datasets
- Verify the generated annotations before training your models