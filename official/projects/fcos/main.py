import os

import tensorflow as tf
from official.projects.fcos.model.model import FCOS
from official.projects.fcos.loss import IOULoss

# TPU Detection and Initialization
# TPU Detection and Initialization
try:
    try:
        # Try local first (TPU VM)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        print("Initialized local TPU.")
    except:
        # Fallback to auto-detect (Remote)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Initialized remote TPU.")
    
    # Only connect to cluster if NOT local
    if tpu.master() != 'local':
        print("Connecting to remote cluster...")
        tf.config.experimental_connect_to_cluster(tpu)
    else:
        print("Skipping cluster connection for local TPU.")
        
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
    
except Exception as e:
    print(f"TPU Initialization failed: {e}")
    
    # Explicit GPU Fallback
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s). Using MirroredStrategy for Multi-GPU/Single-GPU acceleration.")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("No TPU or GPU found. Falling back to CPU default strategy.")
        strategy = tf.distribute.get_strategy()

print("Replica count:", strategy.num_replicas_in_sync)

# Check for Kaggle Dataset Input
# Expected structure: /kaggle/input/coco2014
KAGGLE_DATASET_PATH = "/kaggle/input/coco2014/COCO2014"

# Fallback or alternate name
if not os.path.exists(KAGGLE_DATASET_PATH):
     KAGGLE_DATASET_PATH = "/kaggle/input/coco-2014-downloader"

# Robust Path Detection
def find_path(root, target, is_dir=True):
    for dirpath, dirnames, filenames in os.walk(root):
        if is_dir:
            if target in dirnames:
                return os.path.join(dirpath, target)
        else:
            if target in filenames:
                return os.path.join(dirpath, target)
    return None

if os.path.exists(KAGGLE_DATASET_PATH):
    print(f"Dataset root found at {KAGGLE_DATASET_PATH}")
    
    # Search for Annotations File
    annot_file = find_path(KAGGLE_DATASET_PATH, "instances_train2014.json", is_dir=False)
    if annot_file:
         ANNOTATIONS_PATH = annot_file
         print(f"Found annotations at: {ANNOTATIONS_PATH}")
    else:
         print("WARNING: Could not find instances_train2014.json")
         # Fallback default
         ANNOTATIONS_PATH = os.path.join(KAGGLE_DATASET_PATH, "annotations/instances_train2014.json")

    # Search for Train Images
    train_dir = find_path(KAGGLE_DATASET_PATH, "train2014", is_dir=True)
    if train_dir:
         TRAIN_IMGS_PATH = train_dir + "/" # Ensure trailing slash if code expects it
         print(f"Found train images at: {TRAIN_IMGS_PATH}")
    else:
         print("WARNING: Could not find train2014 directory")
         TRAIN_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "train2014/")

    # Search for Val Images
    val_dir = find_path(KAGGLE_DATASET_PATH, "val2014", is_dir=True)
    if val_dir:
         VAL_IMGS_PATH = val_dir + "/"
    else:
         VAL_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "val2014/")

else:
    # Download COCO 2014 Dataset (if not present locally)
    if not os.path.exists("train2014.zip") and not os.path.exists("train2014"):
        print("Downloading COCO 2014 Dataset...")
        os.system("wget -q http://images.cocodataset.org/zips/train2014.zip")
        os.system("wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
        print("Unzipping...")
        os.system("unzip -q train2014.zip")
        os.system("unzip -q annotations_trainval2014.zip")
        print("Data setup complete.")

    # Paths (Adjusted for Downloaded/Local Data)
    TRAIN_IMGS_PATH = "train2014/"
    VAL_IMGS_PATH = "val2014/" 
    ANNOTATIONS_PATH = "annotations/instances_train2014.json"

# Hyperparameters
BATCH_SIZE = 16 * strategy.num_replicas_in_sync  # Global batch size
EPOCHS = 12  # Standard for 1x schedule, adjust as needed

# Limit dataset for faster debugging/testing if needed
MAX_IMAGES = os.environ.get("MAX_IMAGES", None)
if MAX_IMAGES:
    MAX_IMAGES = int(MAX_IMAGES)
    print(f"Limiting training to {MAX_IMAGES} images.")

def create_learning_rate_schedule():
    # 1x Schedule: divide by 10 at 60k and 80k steps (approx epochs 8 and 11)
    # This needs to be converted to steps based on dataset size
    # For now, using a simple PiecewiseConstantDecay as placeholder
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[60000, 80000],
        values=[0.01, 0.001, 0.0001]
    )

from official.projects.fcos.Data import load_data

with strategy.scope():
    # Model creation must happen inside the strategy scope
    model = FCOS()
    
    # Optimizer
    lr_schedule = create_learning_rate_schedule()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Loss Definitiobs
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
    iou_loss = IOULoss() # Ensure this custom loss is compatible with distributed training
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    
    # Compilation
    model.compile(
        optimizer=optimizer,
        loss={
            'classifier': focal_loss, 
            'box': iou_loss, 
            'centerness': bce_loss
        },
        metrics={'classifier': 'accuracy', 'centerness': 'accuracy'}
    )

def get_dataset(batch_size):
    return load_data.get_training_dataset(TRAIN_IMGS_PATH, ANNOTATIONS_PATH, batch_size, max_images=MAX_IMAGES)

# Training Loop
train_dataset = get_dataset(BATCH_SIZE)
if train_dataset:
     model.fit(
         train_dataset,
         epochs=EPOCHS
     )
print("Model initialized and compiled. Ready for Data Pipeline.")