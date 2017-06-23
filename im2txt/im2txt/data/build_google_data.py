# coding=utf-8
''' build TFrecords, overwrite old ones
# build model, train for one epoch '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image
import logging
import time
import build_mscoco_data
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from im2txt import utils
from collections import namedtuple
from icrawler.builtin import GoogleImageCrawler


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("google_file_folder", "/mnt/raid/data/ni/dnn/zlian/Google_image/",
                       "Directory for saving images returned by Google.")
ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])
vocab_file = "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli/word_counts_copy.txt"
ckpt_foler = "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli"
coco_folder = "/mnt/raid/data/ni/dnn/zlian/mscoco/"
flag_file = "/mnt/raid/data/ni/dnn/zlian/Google_image/flag.txt"
current_folder = None


def predict_images(filenames,vocab, ckpt, n_sentences =2):
    """filenames: list of filenames from disk to infer
       n_sentence: number of sentences generated for each iamge, max=3
       return: list of captions predicted by the most recent ckpt. Each caption shall be a string
       eg: predict_seqs = [["I", "wish","to","get","rid","of","acne"],[******]]
       The real captions to be used in Imagemetadata is different.
       captions=[[u"<S>","I", "wish","to","get","rid","of","acne",".","</S>"]]
    """
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   ckpt_foler)
    g.finalize()

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)
        generator = caption_generator.CaptionGenerator(model, vocab)
        predict_seqs = []
        for filename in filenames:
            with tf.gfile.GFile(filename, "r") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            if (len(captions)< n_sentences):
                n_sentences = len(captions)
            for i in range(n_sentences):
                # Ignore begin and end words.   sentence is a list
                sentence = [vocab.id_to_word(w) for w in captions[i].sentence[1:-2]]
                sentence = " ".join(sentence)
                predict_seqs.append(sentence)
        global_step = model.global_stepp.eval()

    global current_folder
    current_folder =FLAGS.google_file_folder + str(global_step) +'/'
    utils.createfolder(current_folder)
    savingname= current_folder+ 'pred_seqs.pkl'
    utils.save(predict_seqs, savingname, ('Predicted seqs are saved to %s :D') % savingname)
    print ('total number of pred_seqs: %d' %len(predict_seqs))
    return savingname


def png2jpg(png_files):
    for png in png_files:
        im = Image.open(png)
        rgb_im = im.convert('RGB')
        rgb_im.save(png[:-4]+'.jpg')
        os.remove(png)
        print (png + ' is saaaaaaaved :D')
def jPG2jpg(jPG_files):
    for jPG in jPG_files:
        os.rename(jPG, jPG[:-4]+'.jpg')
        print(jPG + ' is saaaaaaaved :D')

def crawl_images(queries, n_google):
    """Queries: A list of strings used to google images.
    n_google: return top n results from google"""
    image_metadata = []
    counter = 0
    for query in queries:
        try:
            counter += 1
            current_dir = current_folder + str(counter)
            google_crawler = GoogleImageCrawler(parser_threads=4, downloader_threads=4,
                                                log_level=logging.ERROR,
                                                storage={'root_dir':current_dir})
            google_crawler.crawl(keyword=query, offset=0, max_num=n_google,
                                 date_min=None, date_max=None,
                                 min_size=(200, 200), max_size=None)
            query = query.split()
            pngs = glob.glob(current_dir + "/*.png")+ glob.glob(current_dir + "/*.PNG")
            JPGs = glob.glob(current_dir + "/*.JPG") + glob.glob(current_dir + "/*.JPEG")
            png2jpg(png_files=pngs)
            jPG2jpg(jPG_files=JPGs)
            images = glob.glob(current_dir + "/*.*jpg")
            for image in images:
                captions = [["<S>"]+query+[".","</S>"]]
                image_metadata.append(ImageMetadata(counter, image, captions))
            # Save metadata every 100 record :D
            if not counter%100:
                utils.save(image_metadata, current_folder+ 'metadata_template_%d.pkl'%counter, '\n')
        except:
            print ('Abandon folder %s' %current_dir)

    print ('Metadata len %d' %len(image_metadata))
    savingname = current_folder+ 'metadata.pkl'
    utils.save(image_metadata, savingname, '\n')
    return savingname


def clear(file_folder):
    # TODO: update clear up function
    os.remove(file_folder, "train-?????-of-00001")

def main():
    # TODO: change this back to 610
    n_infer = 610
    n_google = 10
    n_sentences = 3
    print (n_infer, n_google, n_sentences)

    input_file_folder = "/mnt/raid/data/ni/dnn/zlian/mscoco/raw-data/train2014/"
    train_filenames = glob.glob(input_file_folder + "/*.jpg")
    rand = np.random.randint(len(train_filenames), size=n_infer)
    images_rand = [train_filenames[i] for i in rand]
    ckptpath = glob.glob(ckpt_foler + "/")
    # tf sess exist: flag=1, otherwise, flag=0
    while True:
        flag = utils.readflag(path=flag_file)
        if not flag:
            utils.writeflag(path=flag_file, flag=1, info='start prediction')
            # creating vocab uses tf sess
            vocab = vocabulary.Vocabulary(vocab_file)
            seqpath = predict_images(filenames=images_rand, vocab = vocab, n_sentences=n_sentences, ckpt=ckptpath)
            utils.writeflag(path=flag_file, flag=0, info='finish prediction')
            predict_seqs = utils.load(seqpath, 'Predicted seqs are loaded from %s' % seqpath)
            print ('len of predicted_seqs %s' %len(predict_seqs))
            # This part is vulnerable. Shall change
            metapath = crawl_images(predict_seqs, n_google = n_google)
            metadata = utils.load(metapath, 'Metadata is loaded from %s' % metapath)

            while True:
                flag = utils.readflag(path=flag_file)
                time.sleep(60)
                if not flag:
                    utils.writeflag(path=flag_file, flag=1, info='build data')
                    break
            # build_mscoco_data.output_dir = current_folder
            build_mscoco_data._process_dataset("train", metadata, vocab, num_shards=8)
            try:
                os.system('rm %strain-*****-of-00008' % (coco_folder))
                print ('removed :D')
            except:
                pass
            os.system('cp %s/train* %s' % (FLAGS.google_file_folder, current_folder))
            os.system('cp %s/train* %s' % (FLAGS.google_file_folder, coco_folder))
            os.system('rm %s/train*'%FLAGS.google_file_folder)
            utils.writeflag(path=flag_file, flag=0, info='New images are ready for a new training')
        #     TODO: copy everything in train_dir to ckpt folder :D
        else:
            time.sleep(300)
if __name__=='__main__':
    main()

