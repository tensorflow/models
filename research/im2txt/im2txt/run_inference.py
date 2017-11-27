# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
import pymysql
import re
import time
import socket
from urllib import request
from io import BytesIO
from PIL import Image
import numpy as np

mysql_host = '119.18.193.226'
mysql_user = 'bfsportsdt'
mysql_pass = '85iwx|qttHsrlxPeyldb'
user_set_last_deal_id = 500000
news_image_table = 'news_image_im2text'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)


def _gain_image_info(conn):
    cur = conn.cursor()
    sql = 'select max(original_id) from ai_image.{}'.format(news_image_table)
    cur.execute(sql)
    data = cur.fetchone()
    if data is not None and len(data) != 0 and data[0] is not None:
        last_max_id = data[0]
    else:
        last_max_id = user_set_last_deal_id

    sql = 'select id, content from sports.news where id > {} order by id limit 1000'.format(int(last_max_id))
    cur.execute(sql)
    all_data = cur.fetchall()
    cur.close()
    if len(all_data) == 0:
        return all_data

    res = []
    for original_id, content in all_data:
        imgs = re.findall(pattern='http://image.sports.baofeng.com/[\w]+', string=content)
        res.append((original_id, imgs))
    return res


def get_mysql_conn():
    return pymysql.connect(
        host=mysql_host,
        user=mysql_user,
        passwd=mysql_pass,
        charset='utf8'
    )


# help code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def myCommit(conn, sql, commitList):
    index = 0
    cur = conn.cursor()
    while index < len(commitList):
        l = commitList[index:index + 100]
        index += len(l)
        print("myCommit: sql = ", sql)
        print("myCommit: l = ", l)
        cur.executemany(sql, l)
        conn.commit()
    cur.close()


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    # filenames = []
    # for file_pattern in FLAGS.input_files.split(","):
    #     filenames.extend(tf.gfile.Glob(file_pattern))
    # tf.logging.info("Running caption generation on %d files matching %s", len(filenames), FLAGS.input_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        count = 0
        if True:

            commit_sql = 'replace into ai_image.{} (image_md5, original_id, text1, text2,text3) ' \
                         'values (%s, %s, %s, %s,%s)'.format(news_image_table)

            conn = get_mysql_conn()
            info_list = _gain_image_info(conn)
            conn.close()
            total_start_time = time.time()
            save_info = []
            # info_list = [["im2txt/data/COCO_val2014_000000224477.jpg"]]
            print("info_list size = ", len(info_list))
            for original_id, image_list in info_list:
                if image_list is None or len(image_list) == 0:
                    continue
                for image_path in image_list:
                    print("open =>>>>>>>>>>>> ", image_path)
                    if image_path.startswith('http://'):
                        socket.setdefaulttimeout(2)
                        r = request.urlopen(image_path)
                        image_data = r.read()  # 采用StringIO直接图片文件写到内存，省去写入硬盘
                    else:
                        count = count + 1
                        with tf.gfile.FastGFile(image_path, "rb") as f:
                            image_data = f.read()

                    captions = generator.beam_search(sess, image_data)
                    print("Captions for image %s:" % os.path.basename(image_path))
                    sentence_map = {}
                    for i, caption in enumerate(captions):
                        # Ignore begin and end words.
                        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                        sentence = " ".join(sentence)
                        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                        text = "p=%f : %s " % (math.exp(caption.logprob), sentence)
                        sentence_map.setdefault(i, text)

                    image_md5 = os.path.basename(image_path)
                    save_info.append(
                        (image_md5, original_id, sentence_map.get(0), sentence_map.get(1), sentence_map.get(2)))
                    if len(save_info) > 10:
                        conn = get_mysql_conn()
                        myCommit(conn, commit_sql, save_info)
                        conn.close()
                        save_info = []
                else:
                    conn = get_mysql_conn()
                    myCommit(conn, commit_sql, save_info)
                    conn.close()
                    save_info = []


if __name__ == "__main__":
    tf.app.run()
