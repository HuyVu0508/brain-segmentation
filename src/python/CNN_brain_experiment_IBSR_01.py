from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import math
import h5py


# This script training for data that  have small shape in brain

## Load data has been processing

path='/home/ubuntu/PycharmProjects/HuyCode/Model';
os.chdir(path)

path_model='/home/ubuntu/PycharmProjects/HuyCode/Model/New model/Model';

pickle_file_train = 'Train_Validation_Dataset_4_7.h5'


# part = 0
# offset = 0
with h5py.File(pickle_file_train, 'r') as hf:
    print('List of arrays in this file: \n')
    train_dataset = hf['train_dataset']
    # train_dataset = train_dataset[0:int(round(train_dataset.shape[0]*part))+offset, :]
    # train_dataset = train_dataset[int(round(train_dataset.shape[0]*part)):int(round(train_dataset.shape[0]*1)), :]
    train_dataset = train_dataset[0:train_dataset.shape[0]-1, :]
    train_labels = hf['train_labels']
    # train_labels = train_labels[0:int(round(train_labels.shape[0]*part))+offset, :]

    # train_labels = train_labels[int(round(train_labels.shape[0]*part)):int(round(train_labels.shape[0]*1)), :]
    train_labels = train_labels[0:train_labels.shape[0] - 1, :]
    valid_dataset = hf['valid_dataset']
    # valid_dataset = valid_dataset[0:int(round(valid_dataset.shape[0]*part))+offset,:]
    # valid_dataset = valid_dataset[int(round(valid_dataset.shape[0] * part)) :int(round(valid_dataset.shape[0] * 1)), :]
    valid_dataset = valid_dataset[0:valid_dataset.shape[0]-1, :]
    valid_labels =hf['valid_labels']
    # valid_labels = valid_labels[0:int(round(valid_labels.shape[0]*part))+offset, :]
    # valid_labels = valid_labels[int(round(valid_labels.shape[0]*part)):int(round(valid_labels.shape[0]*1)), :]
    valid_labels = valid_labels[0:valid_labels.shape[0]-1, :]
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    del hf


## Create model


def accuracy(predictions, labels):
  print(predictions.shape)
  print(np.count_nonzero(predictions[:,0]))
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


image_size = 11
num_channels = 1 # grayscale
num_ft=1092
batch_size = 150
patch_size = 2
depth_1 = 13
depth_2 = 26
depth_3 = 39
num_hidden_1= 574
num_hidden_2 = 300
num_hidden_3 = 50
num_labels=2
graph = tf.Graph()

# with graph.as_default():
#
#   # Input data.
#   tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_ft))
#   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#   tf_valid_dataset = tf.constant(valid_dataset)
#
#
#
#   ## Variables.8
#   prob = tf.placeholder(tf.float32)
#
#     # Weightes for CNN 1
#   layer1_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
#   layer1_biases = tf.Variable(tf.zeros([depth_1]))
#
#   layer2_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
#   layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))
#
#   layer3_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
#   layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))
#
#   # Weightes for CNN 2
#
#   layer4_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
#   layer4_biases = tf.Variable(tf.zeros([depth_1]))
#
#   layer5_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
#   layer5_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))
#
#   layer6_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
#   layer6_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))
#
#   # Weightes for CNN 3
#
#   layer7_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
#   layer7_biases = tf.Variable(tf.zeros([depth_1]))
#
#   layer8_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
#   layer8_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))
#
#   layer9_weights = tf.Variable(tf.truncated_normal(
#     [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
#   layer9_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))
#
#   # Weighes for fully connected
#   layer10_weights = tf.Variable(tf.truncated_normal(
#     [1056, num_hidden_1], stddev=math.sqrt(1.0 / num_hidden_1)))
#
#   layer10_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_1]))
#
#   layer11_weights = tf.Variable(tf.truncated_normal(
#     [num_hidden_1, num_hidden_2], stddev=math.sqrt(1.0 / num_hidden_2)))
#
#   layer11_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_2]))
#
#   # layer12_weights = tf.Variable(tf.truncated_normal(
#   #   [num_hidden_2, num_hidden_3], stddev=math.sqrt(1.0 / num_hidden_3)))
#
#   # layer12_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_2]))
#
#   layero_weights = tf.Variable(tf.truncated_normal(
#     [num_hidden_2, num_labels], stddev=0.1))
#   layero_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
#
#
#   # Model.
#   def model(data, prob):
#     data_position = data[:, 363:366]
#     data3_images = data[:, :363]
#
#     data_1_images = data3_images[:, 0:121]
#
#     data_2_images = data3_images[:, 121:242]
#
#     data_3_images = data3_images[:, 242:363]
#
#   # CNN for images 1.
#     shape_data_images_1 = data_1_images.get_shape().as_list()
#     data_1_images = tf.reshape(data_1_images, [shape_data_images_1[0], image_size, image_size, num_channels])
#     conv = tf.nn.conv2d(data_1_images, layer1_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer1_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv = tf.nn.conv2d(conv_pull, layer2_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer2_biases)
#
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     conv = tf.nn.conv2d(conv_pull, layer3_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer3_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     hidden = conv_pull
#
#     shape = hidden.get_shape().as_list()
#     reshape_1 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#
#
#     print(reshape_1)
#
#     # CNN for images 2.
#     shape_data_images_2 = data_2_images.get_shape().as_list()
#     data_2_images = tf.reshape(data_2_images, [shape_data_images_2[0], image_size, image_size, num_channels])
#     conv = tf.nn.conv2d(data_2_images, layer4_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer4_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv = tf.nn.conv2d(conv_pull, layer5_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer5_biases)
#
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     conv = tf.nn.conv2d(conv_pull, layer6_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer6_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     hidden = conv_pull
#
#     shape = hidden.get_shape().as_list()
#     reshape_2 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#
#
#     print(reshape_2)
#
#     # CNN for images 3.
#     shape_data_images_3 = data_3_images.get_shape().as_list()
#     data_3_images = tf.reshape(data_3_images, [shape_data_images_3[0], image_size, image_size, num_channels])
#     conv = tf.nn.conv2d(data_3_images, layer7_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer7_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv = tf.nn.conv2d(conv_pull, layer8_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer8_biases)
#
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     conv = tf.nn.conv2d(conv_pull, layer9_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer9_biases)
#     conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
#
#     hidden = conv_pull
#
#     shape = hidden.get_shape().as_list()
#     reshape_3 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#
#     print(reshape_3)
#
#     ## Merge feature
#
#     reshape = tf.concat(1, [reshape_1,reshape_2,reshape_3, data_position])
#     print(reshape)
#     print(layer10_biases.get_shape())
#     print(layer10_weights.get_shape())
#     reshape
#     hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer10_weights) + layer10_biases), prob)
#     hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, layer11_weights) + layer11_biases), prob)
#     # hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, layer12_weights) + layer12_biases), prob)
#     return (tf.matmul(hidden, layero_weights) + layero_biases)
#
#   # Training computation.
#   logits = model(tf_train_dataset,0.5)
#   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
#
#   # Using the L2-regularization
#   regularizers = (tf.nn.l2_loss(layer1_weights ) + tf.nn.l2_loss(layer1_biases )
#                   + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases)
#                   + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)
#                   + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)
#                   + tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(layer5_biases)
#                   + tf.nn.l2_loss(layer6_weights) + tf.nn.l2_loss(layer6_biases)
#                   + tf.nn.l2_loss(layer7_weights) + tf.nn.l2_loss(layer7_biases)
#                   + tf.nn.l2_loss(layer8_weights) + tf.nn.l2_loss(layer8_biases)
#                   + tf.nn.l2_loss(layer9_weights) + tf.nn.l2_loss(layer9_biases)
#                   + tf.nn.l2_loss(layer10_weights) + tf.nn.l2_loss(layer10_biases)
#                   + tf.nn.l2_loss(layer11_weights) + tf.nn.l2_loss(layer11_biases)
#                   + tf.nn.l2_loss(layero_weights) + tf.nn.l2_loss(layero_biases))
#
#   # # Update Loss with L2-regularization
#   # loss += 5e-4 * regularizers
#
#   # Optimizer.
#   global_step = tf.Variable(0)
#   optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
#
#   # Predictions for the training, validation, and test data.
#   train_prediction = tf.nn.softmax(logits)
#
#   valid_prediction = tf.nn.softmax(model(tf_valid_dataset,1))
#
#
#
#   num_steps = 1200000
#   saver = tf.train.Saver()
#
#   pathsavemodel = "/home/ubuntu/PycharmProjects/HuyCode/Model/"
#   os.chdir(pathsavemodel)
#
#   import time
#   start = time.time()
#   with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     # saver.restore(session, 'Model_IBSR_01-1200000')
#     print('Initialized')
#     for step in range(num_steps):
#       offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#       batch_data = train_dataset[offset:(offset + batch_size), :]
#       batch_labels = train_labels[offset:(offset + batch_size), :]
#       feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#       _, l, predictions = session.run(
#         [optimizer, loss, train_prediction], feed_dict=feed_dict)
#       if (step % 10000 == 0):
#         print('Minibatch loss at step %d: %f' % (step, l))
#         # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#         print('Validation accuracy: %.1f%%' % accuracy(
#           valid_prediction.eval(), valid_labels))
#         # save_path = saver.save(session, 'Model_IBSR_01', global_step=step + 1)
#     save_path = saver.save(session, 'Model_IBSR_01', global_step=step + 1)
#     end = time.time()
#     running_time = end -start
#     print("Time running: %f" % running_time)
#
#
#
#   # with tf.Session(graph=graph) as session:
#   #   saver.restore(session,  'model_brain_CNN_small_2_v1.ckpt-400000')
#   #   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))fh5py.f



with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_ft))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)



  ## Variables.
  prob = tf.placeholder(tf.float32)

    # Weightes for CNN 1
  layer1_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth_1]))

  layer2_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer3_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 2

  layer4_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer4_biases = tf.Variable(tf.zeros([depth_1]))

  layer5_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer6_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer6_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 3

  layer7_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer7_biases = tf.Variable(tf.zeros([depth_1]))

  layer8_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer8_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer9_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer9_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 4

  layer10_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer10_biases = tf.Variable(tf.zeros([depth_1]))

  layer11_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer11_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer12_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer12_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 5

  layer13_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer13_biases = tf.Variable(tf.zeros([depth_1]))

  layer14_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer14_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer15_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer15_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 6

  layer16_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer16_biases = tf.Variable(tf.zeros([depth_1]))

  layer17_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer17_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer18_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer18_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 7

  layer19_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer19_biases = tf.Variable(tf.zeros([depth_1]))

  layer20_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer20_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer21_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer21_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 8

  layer22_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer22_biases = tf.Variable(tf.zeros([depth_1]))

  layer23_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer23_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer24_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer24_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weightes for CNN 9

  layer25_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  layer25_biases = tf.Variable(tf.zeros([depth_1]))

  layer26_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  layer26_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

  layer27_weights = tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_2, depth_3], stddev=0.1))
  layer27_biases = tf.Variable(tf.constant(1.0, shape=[depth_3]))

  # Weighes for fully connected
  layer28_weights = tf.Variable(tf.truncated_normal(
    [3162, num_hidden_1], stddev=math.sqrt(1.0 / num_hidden_1)))
# 1055
  layer28_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_1]))

  layer29_weights = tf.Variable(tf.truncated_normal(
    [num_hidden_1, num_hidden_2], stddev=math.sqrt(1.0 / num_hidden_2)))

  layer29_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_2]))

  layer30_weights = tf.Variable(tf.truncated_normal(
    [num_hidden_2, num_hidden_3], stddev=math.sqrt(1.0 / num_hidden_3)))

  layer30_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_3]))

  layero_weights = tf.Variable(tf.truncated_normal(
    [num_hidden_3, num_labels], stddev=0.1))
  layero_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


  # Model.
  def model(data, prob):
    data_position = data[:, 1089:]
    data3_images = data[:, :1089]

    data_1_images = data3_images[:, 0:121]

    data_2_images = data3_images[:, 121:242]

    data_3_images = data3_images[:, 242:363]

    data_4_images = data3_images[:, 363:484]

    data_5_images = data3_images[:, 484:605]

    data_6_images = data3_images[:, 605:726]

    data_7_images = data3_images[:, 726:847]

    data_8_images = data3_images[:, 847:968]

    data_9_images = data3_images[:, 968:1089]

  # CNN for images 1.
    shape_data_images_1 = data_1_images.get_shape().as_list()
    data_1_images = tf.reshape(data_1_images, [shape_data_images_1[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_1_images, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer3_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer3_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_1 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])


    print(reshape_1)

    # CNN for images 2.
    shape_data_images_2 = data_2_images.get_shape().as_list()
    data_2_images = tf.reshape(data_2_images, [shape_data_images_2[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_2_images, layer4_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer4_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer5_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer5_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer6_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer6_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_2 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])


    print(reshape_2)

    # CNN for images 3.
    shape_data_images_3 = data_3_images.get_shape().as_list()
    data_3_images = tf.reshape(data_3_images, [shape_data_images_3[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_3_images, layer7_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer7_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer8_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer8_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer9_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer9_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_3 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_3)

    # CNN for images 4.
    shape_data_images_4 = data_4_images.get_shape().as_list()
    data_4_images = tf.reshape(data_4_images, [shape_data_images_4[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_4_images, layer10_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer10_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer11_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer11_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer12_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer12_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_4 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_4)

    # CNN for images 5.
    shape_data_images_5 = data_5_images.get_shape().as_list()
    data_5_images = tf.reshape(data_5_images, [shape_data_images_5[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_5_images, layer13_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer13_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer14_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer14_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer15_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer15_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_5 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_5)

    # CNN for images 6.
    shape_data_images_6 = data_6_images.get_shape().as_list()
    data_6_images = tf.reshape(data_6_images, [shape_data_images_6[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_6_images, layer16_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer16_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer17_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer17_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer18_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer18_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_6 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_6)

    # CNN for images 7.
    shape_data_images_7 = data_7_images.get_shape().as_list()
    data_7_images = tf.reshape(data_7_images, [shape_data_images_7[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_7_images, layer19_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer19_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer20_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer20_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer21_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer21_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_7 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_7)

    # CNN for images 8.
    shape_data_images_8 = data_8_images.get_shape().as_list()
    data_8_images = tf.reshape(data_8_images, [shape_data_images_8[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_8_images, layer22_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer22_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer23_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer23_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer24_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer24_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_8 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_8)

    # CNN for images 9.
    shape_data_images_9 = data_9_images.get_shape().as_list()
    data_9_images = tf.reshape(data_9_images, [shape_data_images_9[0], image_size, image_size, num_channels])
    conv = tf.nn.conv2d(data_9_images, layer25_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer25_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(conv_pull, layer26_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer26_biases)

    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(conv_pull, layer27_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer27_biases)
    conv_pull = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    hidden = conv_pull

    shape = hidden.get_shape().as_list()
    reshape_9 = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    print(reshape_9)

    ## Merge feature

    reshape = tf.concat(1, [reshape_1,reshape_2,reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, data_position])
    print(reshape)
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer28_weights) + layer28_biases), prob)
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, layer29_weights) + layer29_biases), prob)
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, layer30_weights) + layer30_biases), prob)
    return (tf.matmul(hidden, layero_weights) + layero_biases)

  # Training computation.
  # session = tf.InteractiveSession()
  logits = model(tf_train_dataset, 0.5)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  tf.scalar_summary("loss", loss)
  #merged = tf.merge_all_summaries()
  #writer = tf.train.SummaryWriter("logs/", session.graph)
  # Using the L2-regularization
  # regularizers = (tf.nn.l2_loss(layer1_weights ) + tf.nn.l2_loss(layer1_biases )
  #                 + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases)
  #                 + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)
  #                 + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)
  #                 + tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(layer5_biases)
  #                 + tf.nn.l2_loss(layer6_weights) + tf.nn.l2_loss(layer6_biases)
  #                 + tf.nn.l2_loss(layer7_weights) + tf.nn.l2_loss(layer7_biases)
  #                 + tf.nn.l2_loss(layero_weights) + tf.nn.l2_loss(layero_biases))

  # # Update Loss with L2-regularization
  # loss += 5e-4 * regularizers

  # Optimizer.
  global_step = tf.Variable(0)
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset,1))



  num_steps = 1200000
  saver = tf.train.Saver()

  import time
  start = time.time()
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    os.chdir(path_model)
    saver.restore(session, 'Model_IBSR_12_part_1-1200000')
    print('Initialized')
    for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 1000 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy(
          valid_prediction.eval(), valid_labels))
        #result = session.run(merged, feed_dict=feed_dict)
        #writer.add_summary(result, step)

      if(step % 100000 == 0):
        save_path = saver.save(session, 'Model_IBSR_12_part_2', global_step=step + 1)
    save_path = saver.save(session, 'Model_IBSR_12_part_2', global_step=step + 1)
    end = time.time()
    running_time = end -start
    print("Time running: %f" % running_time)



  # with tf.Session(graph=graph) as session:
  #   saver.restore(session,  'model_brain_CNN_small_2_v1.ckpt-400000')
  #   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))