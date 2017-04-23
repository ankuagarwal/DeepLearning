from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


################ Loading the dataset ################
pickle_file = '../Lesson 1/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


################ Reformat the data ################
image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


################ Network with 2 convolutional layers ################
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))

  layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev = 0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_labels]))

  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(conv + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


################ Running convolutional model ################
num_steps = 1001

with tf.Session(graph = graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized the convolutional model with 2 strides')

  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


################ Problem 1: Convolutional Layers with 2 stride, kernel size = 2 ################
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))

  layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev = 0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_labels]))

  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding = 'SAME')
    maxpool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(maxpool + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding = 'SAME')
    maxpool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(maxpool + layer2_biases)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


################ Running convolutional model ################
num_steps = 1001

with tf.Session(graph = graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized the convolutional model with 2 strides, 2 kernel size')

  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


################ Problem 2: Adding learning rate ################
# Accuracy Decreased by introducing learning rate
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

start_learning_rate = 0.005
decay_steps = 1000
decay_size = 0.95

graph = tf.Graph()

with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape = [depth]))

  layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev = 0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_labels]))


  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(conv + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding = 'SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))

  # Learning Rate
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_size)

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


################ Running convolutional model ################
num_steps = 1001

with tf.Session(graph = graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized the convolutional model with 2 strides, and Learning Rate')

  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))