from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from IPython import get_ipython
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from hashlib import md5
from sklearn.cross_validation import cross_val_score


################ Downloading the dataset ################
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


################ Extracting the dataset ################
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


################ Problem 1 ################

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


################ Problem 2 ################
# The command `%matplotlib inline` will not work in python script. Use jupyter notebook instead
%matplotlib inline
with open(train_datasets[0], 'rb') as f:
        letter_set = pickle.load(f)
plt.imshow(letter_set[0])
plt.title("Char a")


################ Problem 3 ################
for i in range(0,10):
    with open(train_datasets[i], 'rb') as f:
        letter_set = pickle.load(f)
    print('Size of data set ' + str(len(letter_set)))


################ Merge and Prune Dataset ################
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class + tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


################ Randomize Dataset ################
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


################ Problem 4 ################
# The command `%matplotlib inline` will not work in python script. Use jupyter notebook instead
%matplotlib inline
plt.imshow(train_dataset[np.random.randint(0, len(train_dataset))])


################ Save Data ################
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


################ Problem 5 ################
# The command `%time` will not work in python script. Use jupyter notebook instead
%time set_valid_dataset = set([ md5(x).hexdigest() for x in valid_dataset])
%time set_test_dataset = set([ md5(x).hexdigest() for x in test_dataset])
%time set_train_dataset = set([ md5(x).hexdigest() for x in train_dataset])

overlap_test_valid = set_test_dataset - set_valid_dataset
print('overlap test valid: ' + str(len(overlap_test_valid)))

overlap_train_valid = set_train_dataset - set_valid_dataset
print ('overlap train valid: ' + str(len(overlap_train_valid)))

overlap_train_test = set_train_dataset - set_test_dataset
print ('overlap train test: ' + str(len(overlap_train_test)))


################ Sanitize the data ################
with open('notMNIST.pickle', 'rb') as f:
        datasets = pickle.load(f)

def letter(i):
    return 'ABCDEFGHIJ'[i]

all_labels = np.concatenate((datasets['train_labels'],
                             datasets['valid_labels'],
                             datasets['test_labels']))

all_data = np.concatenate((datasets['train_dataset'],
                             datasets['valid_dataset'],
                             datasets['test_dataset']))

sanitized_data = []
sanitized_labels = []

hashes = set()

for i in range(0,len(all_labels)):
    if not md5(all_data[i]).hexdigest() in hashes:
        sanitized_data.append(all_data[i])
        sanitized_labels.append(all_labels[i])
        hashes.add(md5(all_data[i]).hexdigest())

sanitized_data = np.stack(sanitized_data)
sanitized_labels = np.stack(sanitized_labels)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation, :, :]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

sanitized_data, sanitized_labels = randomize(sanitized_data, sanitized_labels)

valid_dataset =  sanitized_data[:10000]
valid_labels = sanitized_labels[:10000]
test_dataset =  sanitized_data[10000:20000]
test_labels = sanitized_labels[10000:20000]
train_dataset = sanitized_data[20000:]
train_labels = sanitized_labels[20000:]

pickle_file = 'notMNIST_Sanitized.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


################ Problem 6 ################
with open('notMNIST_Sanitized.pickle', 'rb') as f:
        datasets_sanitized = pickle.load(f)
with open('notMNIST.pickle', 'rb') as f:
        datasets = pickle.load(f)

logregCV = LogisticRegression()

flat_train_dataset_50 = [x.flatten() for x in datasets['train_dataset'][:50]]
flat_train_dataset_100 = [x.flatten() for x in datasets['train_dataset'][:100]]
flat_train_dataset_1000 = [x.flatten() for x in datasets['train_dataset'][:1000]]
flat_train_dataset_5000 = [x.flatten() for x in datasets['train_dataset'][:5000]]
flat_train_dataset = [x.flatten() for x in datasets['train_dataset']]

flat_train_dataset_50_san = [x.flatten() for x in datasets_sanitized['train_dataset'][:50]]
flat_train_dataset_100_san = [x.flatten() for x in datasets_sanitized['train_dataset'][:100]]
flat_train_dataset_1000_san = [x.flatten() for x in datasets_sanitized['train_dataset'][:1000]]
flat_train_dataset_5000_san = [x.flatten() for x in datasets_sanitized['train_dataset'][:5000]]
flat_train_dataset_san = [x.flatten() for x in datasets_sanitized['train_dataset']]

model = logregCV.fit(flat_train_dataset, datasets['train_labels'])
model_50 = logregCV.fit(flat_train_dataset_50, datasets['train_labels'][:50])
model_100 = logregCV.fit(flat_train_dataset_100, datasets['train_labels'][:100])
model_1000 = logregCV.fit(flat_train_dataset_1000, datasets['train_labels'][:1000])
model_5000 = logregCV.fit(flat_train_dataset_5000, datasets['train_labels'][:5000])

model_sanitized = logregCV.fit(flat_train_dataset_san, datasets_sanitized['train_labels'])
model_50_sanitized = logregCV.fit(flat_train_dataset_50_san, datasets_sanitized['train_labels'][:50])
model_100_sanitized = logregCV.fit(flat_train_dataset_100_san, datasets_sanitized['train_labels'][:100])
model_1000_sanitized = logregCV.fit(flat_train_dataset_1000_san, datasets_sanitized['train_labels'][:1000])
model_5000_sanitized = logregCV.fit(flat_train_dataset_5000_san, datasets_sanitized['train_labels'][:5000])

score = model.score(flat_train_dataset, datasets['train_labels'])
score_50 = model_50.score(flat_train_dataset_50, datasets['train_labels'][:50])
score_100 = model_100.score(flat_train_dataset_100, datasets['train_labels'][:100])
score_1000 = model_1000.score(flat_train_dataset_1000, datasets['train_labels'][:1000])
score_5000 = model_5000.score(flat_train_dataset_5000, datasets['train_labels'][:5000])

score_san = model_sanitized.score(flat_train_dataset_san, datasets['train_labels'])
score_50_san = model_50_sanitized.score(flat_train_dataset_50_san, datasets_sanitized['train_labels'][:50])
score_100_san = model_100_sanitized.score(flat_train_dataset_100_san, datasets_sanitized['train_labels'][:100])
score_1000_san = model_1000_sanitized.score(flat_train_dataset_1000_san, datasets_sanitized['train_labels'][:1000])
score_5000_san = model_5000_sanitized.score(flat_train_dataset_5000_san, datasets_sanitized['train_labels'][:5000])

print("Scores and Mean for normal dataset and sanitized dataset")

print(datasets['train_labels'].mean())
print(score)
print(score_san)

print(datasets_sanitized['train_labels'][:50].mean())
print(score_50)
print(score_50_san)

print(datasets_sanitized['train_labels'][:100].mean())
print(score_100)
print(score_100_san)

print(datasets_sanitized['train_labels'][:1000].mean())
print(score_1000)
print(score_1000_san)

print(datasets_sanitized['train_labels'][:5000].mean())
print(score_5000)
print(score_5000_san)
