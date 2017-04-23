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

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
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


################ Problem 1 ################

################ Loading data for Stochastic Gradient Descent ################
batch_size = 128
beta = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits) +
  			beta * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


################ Running Stochastic Gradient Descent ################
num_steps = 3001

with tf.Session(graph = graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.global_variables_initializer().run()
  print ("Initialized Stochastic Gradient Descent")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


################ Loading data for Neural Network (SGD) ################
batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  num_neurons = 1024

  # Hidden Variables.
  hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_neurons]))
  hidden_biases = tf.Variable(tf.zeros([num_neurons]))

  # Hidden Layer
  hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

  hidden_out_weights = tf.Variable(tf.truncated_normal([num_neurons, num_labels]))
  hidden_out_biases = tf.Variable(tf.zeros([num_labels]))

  # Last Layer
  logits = tf.matmul(hidden_layer, hidden_out_weights) + hidden_out_biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits) +
  			beta * (tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_biases)) +
  			beta * ((tf.nn.l2_loss(hidden_out_weights) + tf.nn.l2_loss(hidden_out_biases))))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_relu = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_relu, hidden_out_weights) + hidden_out_biases)

  test_relu = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu, hidden_out_weights) + hidden_out_biases)


################ Running Neural Network (SGD) ################
num_steps = 3001

with tf.Session(graph = graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.global_variables_initializer().run()
  print ("Initialized Neural Network for SGD")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


################ Problem 2 ################

################ Loading data for Stochastic Gradient Descent ################
batch_size = 64 # Decreasing the batch size to 64
beta = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits) +
  			beta * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


################ Running Stochastic Gradient Descent ################
num_steps = 3001

with tf.Session(graph = graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.global_variables_initializer().run()
  print ("Initialized Stochastic Gradient Descent decreasing batch size")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


################ Problem 3 ################

################# Loading data for Neural Network (SGD) ################
batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  num_neurons = 1024

  # Hidden Variables.
  hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_neurons]))
  hidden_biases = tf.Variable(tf.zeros([num_neurons]))

  # Hidden Layer
  hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

  # Adding dropouts in Hidden layer
  keep_prob = tf.placeholder(tf.float32)
  hidden_layer_dropout = tf.nn.dropout(hidden_layer, keep_prob)

  hidden_out_weights = tf.Variable(tf.truncated_normal([num_neurons, num_labels]))
  hidden_out_biases = tf.Variable(tf.zeros([num_labels]))

  # Last Layer
  logits = tf.matmul(hidden_layer_dropout, hidden_out_weights) + hidden_out_biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits) +
  			beta * (tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_biases)) +
  			beta * ((tf.nn.l2_loss(hidden_out_weights) + tf.nn.l2_loss(hidden_out_biases))))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_relu = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_relu, hidden_out_weights) + hidden_out_biases)

  test_relu = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu, hidden_out_weights) + hidden_out_biases)


################ Running Neural Network (SGD) ################
num_steps = 3001

with tf.Session(graph = graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.global_variables_initializer().run()
  print ("Initialized Neural Network for SGD adding dropouts in hidden layer")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


################ Problem 4 ################

################# Loading data for Neural Network (SGD) ################
batch_size = 128

num_neurons = 1024

start_learning_rate = 0.005
decay_steps = 1000
decay_size = 0.95

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Hidden Variables Layer 1.
  hidden_weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_neurons]))
  hidden_biases_1 = tf.Variable(tf.zeros([num_neurons]))

  # Hidden Layer 1
  hidden_layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights_1) + hidden_biases_1)

  # Adding dropouts in Hidden layer 2
  keep_prob = tf.placeholder(tf.float32)
  hidden_layer_dropout_1 = tf.nn.dropout(hidden_layer_1, keep_prob)

  # Hidden Variables Layer 2.
  hidden_weights_2 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons]))
  hidden_biases_2 = tf.Variable(tf.zeros([num_neurons]))

  # Hidden Layer 2
  hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_dropout_1, hidden_weights_2) + hidden_biases_2)

  # Adding dropouts in Hidden layer 2
  hidden_layer_dropout_2 = tf.nn.dropout(hidden_layer_2, keep_prob)

  hidden_out_weights = tf.Variable(tf.truncated_normal([num_neurons, num_labels]))
  hidden_out_biases = tf.Variable(tf.zeros([num_labels]))

  # Last Layer
  logits = tf.matmul(hidden_layer_dropout_2, hidden_out_weights) + hidden_out_biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits) +
  			beta * (tf.nn.l2_loss(hidden_weights_1) + tf.nn.l2_loss(hidden_biases_1)) +
  			beta * (tf.nn.l2_loss(hidden_weights_2) + tf.nn.l2_loss(hidden_biases_2)) +
  			beta * ((tf.nn.l2_loss(hidden_out_weights) + tf.nn.l2_loss(hidden_out_biases))))


  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_size)

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_relu_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights_1) + hidden_biases_1)
  valid_relu_2 = tf.nn.relu(tf.matmul(valid_relu_1, hidden_weights_2) + hidden_biases_2)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_relu_2, hidden_out_weights) + hidden_out_biases)

  test_relu_1 = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights_1) + hidden_biases_1)
  test_relu_2 = tf.nn.relu(tf.matmul(test_relu_1, hidden_weights_2) + hidden_biases_2)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu_2, hidden_out_weights) + hidden_out_biases)


################ Running Neural Network (SGD) ################
num_steps = 3001

with tf.Session(graph = graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.global_variables_initializer().run()
  print ("Initialized Neural Network for SGD adding dropouts in 2 hidden layer and learing rate")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

