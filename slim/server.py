# coding=utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import time
from flask import request, send_from_directory
from flask import Flask, request, redirect, url_for
import uuid
import tensorflow as tf
from classify_image import run_inference_on_image

ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('model_name', 'my_inception_v4_freeze.pb', '')
tf.app.flags.DEFINE_string('label_file', 'my_inception_v4_freeze.label', '')
tf.app.flags.DEFINE_string('upload_folder', '/tmp/', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '5001',
        'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = FLAGS.upload_folder
ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER

def allowed_files(filename):
  return '.' in filename and \
      filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_file_name):
  basename = os.path.basename(old_file_name)
  name, ext = os.path.splitext(basename)
  new_name = str(uuid.uuid1()) + ext
  return new_name

def inference(file_name):
  try:
    predictions, top_k, top_names = run_inference_on_image(file_name, model_file=FLAGS.model_name)
    print(predictions)
  except Exception as ex: 
    print(ex)
    return ""
  new_url = '/static/%s' % os.path.basename(file_name)
  image_tag = '<img src="%s"></img><p>'
  new_tag = image_tag % new_url
  format_string = ''
  for node_id, human_name in zip(top_k, top_names):
    score = predictions[node_id]
    format_string += '%s (score:%.5f)<BR>' % (human_name, score)
  ret_string = new_tag  + format_string + '<BR>' 
  return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
  result = """
    <!doctype html>
    <title>临时测试用</title>
    <h1>来喂一张照片吧</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
         <input type=submit value='上传'>
    </form>
    <p>%s</p>
    """ % "<br>"
  if request.method == 'POST':
    file = request.files['file']
    old_file_name = file.filename
    if file and allowed_files(old_file_name):
      filename = rename_filename(old_file_name)
      file_path = os.path.join(UPLOAD_FOLDER, filename)
      file.save(file_path)
      type_name = 'N/A'
      print('file saved to %s' % file_path)
      out_html = inference(file_path)
      return result + out_html 
  return result

if __name__ == "__main__":
  print('listening on port %d' % FLAGS.port)
  app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)

