from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import input_data
import pathnet

import numpy as np
import time

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  total_tr_data, total_tr_label = mnist.train.next_batch(mnist.train._num_examples);
  
  # Gathering a1 Data
  tr_data_a1=total_tr_data[(total_tr_label[:,FLAGS.a1]==1.0)];
  for i in range(len(tr_data_a1)):
    for j in range(len(tr_data_a1[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_a1[i,j]=np.minimum(tr_data_a1[i,j]+rand_num,1.0);
  
  # Gathering a2 Data
  tr_data_a2=total_tr_data[(total_tr_label[:,FLAGS.a2]==1.0)];
  for i in range(len(tr_data_a2)):
    for j in range(len(tr_data_a2[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_a2[i,j]=np.minimum(tr_data_a2[i,j]+rand_num,1.0);
  
  # Gathering b1 Data
  tr_data_b1=total_tr_data[(total_tr_label[:,FLAGS.b1]==1.0)];
  for i in range(len(tr_data_b1)):
    for j in range(len(tr_data_b1[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_b1[i,j]=np.minimum(tr_data_b1[i,j]+rand_num,1.0);

  # Gathering b2 Data
  tr_data_b2=total_tr_data[(total_tr_label[:,FLAGS.b2]==1.0)];
  for i in range(len(tr_data_b2)):
    for j in range(len(tr_data_b2[0])):
      rand_num=np.random.rand();
      if(rand_num>=0.5):
        tr_data_b2[i,j]=np.minimum(tr_data_b2[i,j]+rand_num,1.0);

  tr_data1=np.append(tr_data_a1,tr_data_a2,axis=0);
  tr_label1=np.zeros((len(tr_data1),2),dtype=float);
  for i in range(len(tr_data1)):
    if(i<len(tr_data_a1)):
      tr_label1[i,0]=1.0;
    else:
      tr_label1[i,1]=1.0;

  tr_data2=np.append(tr_data_b1,tr_data_b2,axis=0);
  tr_label2=np.zeros((len(tr_data2),2),dtype=float);
  for i in range(len(tr_data2)):
    if(i<len(tr_data_b1)):
      tr_label2[i,0]=1.0;
    else:
      tr_label2[i,1]=1.0;
  
  ## TASK 1
  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 2)

  # geopath_examples
  geopath=pathnet.geopath_initializer(FLAGS.L,FLAGS.M);
  
  # fixed weights list
  fixed_list=np.ones((FLAGS.L,FLAGS.M),dtype=str);
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      fixed_list[i,j]='0';    

  # Hidden Layers
  weights_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
  biases_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(i==0):
        weights_list[i,j]=pathnet.module_weight_variable([784,FLAGS.filt]);
        biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);
      else:
        weights_list[i,j]=pathnet.module_weight_variable([FLAGS.filt,FLAGS.filt]);
        biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);
  
  for i in range(FLAGS.L):
    layer_modules_list=np.zeros(FLAGS.M,dtype=object);
    for j in range(FLAGS.M):
      if(i==0):
        layer_modules_list[j]=pathnet.module(x, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1))*geopath[i,j];
      else:
        layer_modules_list[j]=pathnet.module2(j,net, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1))*geopath[i,j];
    net=np.sum(layer_modules_list)/FLAGS.M;
  #net=net/FLAGS.M;  
  # Output Layer
  output_weights=pathnet.module_weight_variable([FLAGS.filt,2]);
  output_biases=pathnet.module_bias_variable([2]);
  y = pathnet.nn_layer(net,output_weights,output_biases,'output_layer');

  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)
  
  # Need to learn variables
  var_list_to_learn=[]+output_weights+output_biases;
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if (fixed_list[i,j]=='0'):
        var_list_to_learn+=weights_list[i,j]+biases_list[i,j];
  
  # GradientDescent 
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);

  # Accuracy 
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train1', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test1')
  tf.global_variables_initializer().run()

  # Generating randomly geopath
  geopath_set=np.zeros(FLAGS.candi,dtype=object);
  for i in range(FLAGS.candi):
    geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
  
  # parameters placeholders and ops 
  var_update_ops=np.zeros(len(var_list_to_learn),dtype=object);
  var_update_placeholders=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_update_placeholders[i]=tf.placeholder(var_list_to_learn[i].dtype,shape=var_list_to_learn[i].get_shape());
    var_update_ops[i]=var_list_to_learn[i].assign(var_update_placeholders[i]);
 
  # geopathes placeholders and ops 
  geopath_update_ops=np.zeros((len(geopath),len(geopath[0])),dtype=object);
  geopath_update_placeholders=np.zeros((len(geopath),len(geopath[0])),dtype=object);
  for i in range(len(geopath)):
    for j in range(len(geopath[0])):
      geopath_update_placeholders[i,j]=tf.placeholder(geopath[i,j].dtype,shape=geopath[i,j].get_shape());
      geopath_update_ops[i,j]=geopath[i,j].assign(geopath_update_placeholders[i,j]);
     
  acc_geo=np.zeros(FLAGS.B,dtype=float); 
  summary_geo=np.zeros(FLAGS.B,dtype=object); 
  for i in range(FLAGS.max_steps):
    # Select Candidates to Tournament
    compet_idx=range(FLAGS.candi);
    np.random.shuffle(compet_idx);
    compet_idx=compet_idx[:FLAGS.B];
    # Learning & Evaluating
    for j in range(len(compet_idx)):
      # Shuffle the data
      idx=range(len(tr_data1));
      np.random.shuffle(idx);
      tr_data1=tr_data1[idx];tr_label1=tr_label1[idx];
      # Insert Candidate
      pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[compet_idx[j]],FLAGS.L,FLAGS.M);
      acc_geo_tr=0;
      for k in range(FLAGS.T):
        summary_geo_tr, _, acc_geo_tmp = sess.run([merged, train_step,accuracy], feed_dict={x:tr_data1[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:],y_:tr_label1[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:]});
        acc_geo_tr+=acc_geo_tmp;
      acc_geo[j]=acc_geo_tr/FLAGS.T;
      summary_geo[j]=summary_geo_tr;
    # Tournament
    winner_idx=np.argmax(acc_geo);
    acc=acc_geo[winner_idx];
    summary=summary_geo[winner_idx];
    # Copy and Mutation
    for j in range(len(compet_idx)):
      if(j!=winner_idx):
        geopath_set[compet_idx[j]]=np.copy(geopath_set[compet_idx[winner_idx]]);
        geopath_set[compet_idx[j]]=pathnet.mutation(geopath_set[compet_idx[j]],FLAGS.L,FLAGS.M,FLAGS.N);
    train_writer.add_summary(summary, i);
    print('Training Accuracy at step %s: %s' % (i, acc));
    if(acc >= 0.99):
      print('Learning Done!!');
      print('Optimal Path is as followed.');
      print(geopath_set[compet_idx[winner_idx]]);
      task1_optimal_path=geopath_set[compet_idx[winner_idx]];
      break;
    """
    geopath_sum=np.zeros((len(geopath),len(geopath[0])),dtype=float);
    for j in range(len(geopath_set)):
      for k in range(len(geopath)):
        for l in range(len(geopath[0])):
          geopath_sum[k][l]+=geopath_set[j][k][l];
    print(geopath_sum);
    """    
  iter_task1=i;    
  
  # Fix task1 Optimal Path
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(task1_optimal_path[i,j]==1.0):
        fixed_list[i,j]='1';
  
  # Get variables of fixed list
  var_list_to_fix=[];
  #var_list_to_fix=[]+output_weights+output_biases;
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(fixed_list[i,j]=='1'):
        var_list_to_fix+=weights_list[i,j]+biases_list[i,j];
  var_list_fix=pathnet.parameters_backup(var_list_to_fix);

  """
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(task1_optimal_path[i,j]==1.0):
        fixed_list[i,j]='0';
  """

  # parameters placeholders and ops 
  var_fix_ops=np.zeros(len(var_list_to_fix),dtype=object);
  var_fix_placeholders=np.zeros(len(var_list_to_fix),dtype=object);
  for i in range(len(var_list_to_fix)):
    var_fix_placeholders[i]=tf.placeholder(var_list_to_fix[i].dtype,shape=var_list_to_fix[i].get_shape());
    var_fix_ops[i]=var_list_to_fix[i].assign(var_fix_placeholders[i]);
 
  ## TASK 2
  # Need to learn variables
  var_list_to_learn=[]+output_weights+output_biases;
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if (fixed_list[i,j]=='0'):
        var_list_to_learn+=weights_list[i,j]+biases_list[i,j];
  
  for i in range(FLAGS.L):
    for j in range(FLAGS.M):
      if(fixed_list[i,j]=='1'):
        tmp=biases_list[i,j][0];
        break;
    break;

  # Initialization
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train2', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test2')
  tf.global_variables_initializer().run()
  
  # Update fixed values
  pathnet.parameters_update(sess,var_fix_placeholders,var_fix_ops,var_list_fix);
 
  # GradientDescent  
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);
  
  # Generating randomly geopath
  geopath_set=np.zeros(FLAGS.candi,dtype=object);
  for i in range(FLAGS.candi):
    geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
  
  # parameters placeholders and ops 
  var_update_ops=np.zeros(len(var_list_to_learn),dtype=object);
  var_update_placeholders=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_update_placeholders[i]=tf.placeholder(var_list_to_learn[i].dtype,shape=var_list_to_learn[i].get_shape());
    var_update_ops[i]=var_list_to_learn[i].assign(var_update_placeholders[i]);
  
  acc_geo=np.zeros(FLAGS.B,dtype=float); 
  summary_geo=np.zeros(FLAGS.B,dtype=object); 
  for i in range(FLAGS.max_steps):
    # Select Candidates to Tournament
    compet_idx=range(FLAGS.candi);
    np.random.shuffle(compet_idx);
    compet_idx=compet_idx[:FLAGS.B];
    # Learning & Evaluating
    for j in range(len(compet_idx)):
      # Shuffle the data
      idx=range(len(tr_data2));
      np.random.shuffle(idx);
      tr_data2=tr_data2[idx];tr_label2=tr_label2[idx];
      geopath_insert=np.copy(geopath_set[compet_idx[j]]);
      
      for l in range(FLAGS.L):
        for m in range(FLAGS.M):
          if(fixed_list[l,m]=='1'):
            geopath_insert[l,m]=1.0;
      
      # Insert Candidate
      pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_insert,FLAGS.L,FLAGS.M);
      acc_geo_tr=0;
      for k in range(FLAGS.T):
        summary_geo_tr, _, acc_geo_tmp = sess.run([merged, train_step,accuracy], feed_dict={x:tr_data2[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:],y_:tr_label2[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:]});
        acc_geo_tr+=acc_geo_tmp;
      acc_geo[j]=acc_geo_tr/FLAGS.T;
      summary_geo[j]=summary_geo_tr;
    # Tournament
    winner_idx=np.argmax(acc_geo);
    acc=acc_geo[winner_idx];
    summary=summary_geo[winner_idx];
    # Copy and Mutation
    for j in range(len(compet_idx)):
      if(j!=winner_idx):
        geopath_set[compet_idx[j]]=np.copy(geopath_set[compet_idx[winner_idx]]);
        geopath_set[compet_idx[j]]=pathnet.mutation(geopath_set[compet_idx[j]],FLAGS.L,FLAGS.M,FLAGS.N);
    train_writer.add_summary(summary, i);
    print('Training Accuracy at step %s: %s' % (i, acc));
    if(acc >= 0.99):
      print('Learning Done!!');
      print('Optimal Path is as followed.');
      print(geopath_set[compet_idx[winner_idx]]);
      task2_optimal_path=geopath_set[compet_idx[winner_idx]];
      break;
    """
    geopath_sum=np.zeros((len(geopath),len(geopath[0])),dtype=float);
    for j in range(len(geopath_set)):
      for k in range(len(geopath)):
        for l in range(len(geopath[0])):
          geopath_sum[k][l]+=geopath_set[j][k][l];
    print(geopath_sum);
    """

  iter_task2=i;      
  overlap=0;
  for i in range(len(task1_optimal_path)):
    for j in range(len(task1_optimal_path[0])):
      if(task1_optimal_path[i,j]==task2_optimal_path[i,j])&(task1_optimal_path[i,j]==1.0):
        overlap+=1;
  print("Entire Iter:"+str(iter_task1+iter_task2)+",TASK1:"+str(iter_task1)+",TASK2:"+str(iter_task2)+",Overlap:"+str(overlap));
 
  train_writer.close()
  test_writer.close()


def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--learning_rate', type=float, default=0.05,
                      help='Initial learning rate')
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/pathnet/binary_mnist/pathnet/1_3_1_2',
                      help='Summaries log directry')
  parser.add_argument('--M', type=int, default=10,
                      help='The Number of Modules per Layer')
  parser.add_argument('--L', type=int, default=3,
                      help='The Number of Layers')
  parser.add_argument('--N', type=int, default=3,
                      help='The Number of Selected Modules per Layer')
  parser.add_argument('--T', type=int, default=50,
                      help='The Number of epoch per each geopath')
  parser.add_argument('--batch_num', type=int, default=16,
                      help='The Number of batches per each geopath')
  parser.add_argument('--filt', type=int, default=20,
                      help='The Number of Filters per Module')
  parser.add_argument('--candi', type=int, default=20,
                      help='The Number of Candidates of geopath')
  parser.add_argument('--B', type=int, default=2,
                      help='The Number of Candidates for each competition')
  parser.add_argument('--a1', type=int, default=1,
                      help='The first class of task1')
  parser.add_argument('--a2', type=int, default=3,
                      help='The second class of task1')
  parser.add_argument('--b1', type=int, default=1,
                      help='The first class of task2')
  parser.add_argument('--b2', type=int, default=2,
                      help='The second class of task2')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
