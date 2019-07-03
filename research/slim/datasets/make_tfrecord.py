'''
Create by hjkim in bigdata Lab. KOOKMIN Univ.
'''
import matplotlib.pyplot as plt
from matplotlib.image import imread
'''
This function will return a categories of dataset if you serve the dataset root path.
'''
class MakeTfRecord():
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.dirs = []
        self.image_labels = {}
        '''
        image_labels = { label_1 : 0
                         label_2 : 1
                         ......
                         }
        '''
        
    # The following functions can be used to convert a value to a type compatible
    # with tf.Example.
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _get_categories(self):
        labels = os.listdir(self.image_dir)
        for label in labels:
            data_dir = os.path.join(self.image_dir, label)
            if data_dir.find('checkpoints') > 0:
                os.rmdir(data_dir)
                
            if os.path.isdir(data_dir) == True:
                self.dirs.append(label)
    
    def _create_labels(self):
        #create dictionary for label.
        self._get_categories()
        idx = 0
        for category in self.dirs:
            self.image_labels[category] = idx
            idx += 1
    
    def _generates_data(self):
        '''
        will be return the list
        [path, label]
        '''
        for category in self.dirs:
            print(category)
            ctg_path = os.path.join(self.image_dir, category)
            files = os.listdir(ctg_path)
            
            for file in files:
                path = os.path.join(ctg_path, file)
                yield path, self.image_labels[category]

    # Create a dictionary with features that may be relevant.
    def _image_example(self, im, image_string, label):
        image_shape = im.shape
        feature = {
                    'image/format': self._bytes_feature("png"),
                    'image/class/label': self._int64_feature(label),
                    'image/encoded': self._bytes_feature(image_string),
                }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    

    def _writer_fn(self, sess, c_label, total_category_num):
        image_reader = ImageReader()
        gen = self._generates_data()
        with tf.python_io.TFRecordWriter('/dockermnt/dataset/training_tfrecord/trainImage_{0}_of_{1}.tfrecord'.format(str(c_label), str(len(total_category_num)))) as writer:
            for path, label in gen:
                if c_label == label :
                    # Read the filename:
                    image_data = tf.gfile.GFile(path, 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)


                    example = dataset_utils.image_to_tfexample(
                    image_data, b'png', height, width, label)
                    '''
                    # Why do not get of shape???
                    image_string = open(path, 'rb').read()
                    im = imread(path) # so, i had get shape using the plt lib.
                    tfrecord = self._image_example(im, image_string, label)
                    '''
                    writer.write(example.SerializeToString())

    def build(self):
        '''
		tfrecord_mk = MakeTfRecord()
		tfrecord_mk.build()
		You can see that the log prints a categories in directory.
        '''
        self._create_labels() # label key-map
        total_category_num = os.listdir(self.image_dir)
        print(total_category_num)
        with tf.Session('') as sess:            
            for c_label in range(70): # fixed
                self._writer_fn(sess, c_label, total_category_num)

