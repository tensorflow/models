import urllib.request
import tensorflow as tf
import itertools

URL = 'http://download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001'
DST_ORIG = 'fsns-00000-of-00001.orig'
DST = 'fsns-00000-of-00001'
KEEP_NUM_RECORDS = 5

print('Downloading %s ...' % URL)
urllib.request.urlretrieve(URL, DST_ORIG)

print('Writing %d records from %s to %s ...' %
      (KEEP_NUM_RECORDS, DST_ORIG, DST))
with tf.io.TFRecordWriter(DST) as writer:
    for raw_record in itertools.islice(tf.compat.v1.python_io.tf_record_iterator(DST_ORIG), KEEP_NUM_RECORDS):
        writer.write(raw_record)
