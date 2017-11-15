import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    train = pd.read_csv(train_path, names=COLUMNS, header=0)
    train_x, train_y = train, train.pop(y_name)

    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=COLUMNS, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def datasets(y_name='Species'):
    (train_x, train_y), (test_x, test_y) = load_data(y_name=y_name)

    return (
        make_dataset(dict(train_x), train_y),
        make_dataset(dict(test_x), test_y)
    )

def make_dataset(*inputs):
    return tf.data.Dataset.from_tensor_slices(inputs)


