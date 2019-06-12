# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


# Reads training, validation and testing data from the data folder.

"""Utilities for parsing PTB text files."""
import collections
import os

import tensorflow as tf
import numpy as np


def _read_words(filename):
  with tf.gfile.GFile(filename, "rb") as f:
    return list(f.read())


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)# 输出一个计数的字典
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))# sorted默认从小到大排,此处先按第一维从大到小排，再按第0维从小到大排

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, 'train')
  valid_path = os.path.join(data_path, 'valid')

  ##########################################################################################3
  test_path  = os.path.join(data_path, 'test')

  word_to_id = _build_vocab(train_path)
  print('vocabulary size:', len(word_to_id))
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  print('data loaded')
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    print("epoch_size")
    print(epoch_size)
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y

def kdd_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.float32)#raw data : train_data | vali_data | test data
  data_len = len(raw_data) #how many words in the data_set
  batch_num = data_len//batch_size  #the number of batched 

  print("raw_data")
  print(raw_data.shape)
  i = tf.train.range_input_producer(batch_num, shuffle=False).dequeue()
  #x = tf.slice(raw_data, [i * batch_size,0], [batch_size, 42]) #unsw-nb15
  #y_ = tf.slice(raw_data, [i * batch_size,42], [batch_size, 1]) #unsw-nb15
  x = tf.slice(raw_data, [i * batch_size,0], [batch_size, 5]) #ae30
  y_ = tf.slice(raw_data, [i * batch_size,5], [batch_size, 1]) #ae30

  y = tf.cond(tf.equal(y_[0,0], 0.0), lambda: tf.constant([[1,0]]), lambda: tf.constant([[0,1]]))
  for k in range(batch_size-1):
    y = tf.cond(tf.equal(y_[k+1,0], 0.0), lambda: tf.concat([y, tf.constant([[1,0]])], 0), 
      lambda: tf.concat([y, tf.constant([[0,1]])], 0))
  #y = [1,0] #normal
  #for k in range(batch_size-1):
  #  y = tf.cond(tf.equal(y_[k+1,0], tf.constant(1.0)), lambda: y + [[1,0]], lambda: y + [[0,1]])
  #print(y)



  #for k in range(batch_size):
  #  if tf.equal(y_[k], 1.0):
  #    y.append([1,0])#正常
  #  else:
  #    y.append([0,1])#异常
    #print(y.shape)
  return x, y
  #for j in range(batch_num):
  #  x = raw_data[j*batch_size:(j+1)*batch_size, 0:32]
  #  y = []
  #  for k in range(batch_size):
  #    if raw_data[j*batch_size+k, 32:33] == 1:
  #      y.append([1,0])#正常
  #    else:
  #      y.append([0,1])#异常
  #    #print(y.shape)
  #  yield (x, y)

  #print(j)
  #print(y) 