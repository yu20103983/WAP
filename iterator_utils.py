# coding=utf-8
# Copyright 2017 Google Inc. All Rights Reserved.
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

"""For loading data into NMT models."""
from __future__ import print_function

import collections
from config import cfg

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"
                                           ))):
  pass

def normalize_input_img(img):
    shape = tf.shape(img)

    def f1(shape, img):
        shape = tf.cast(shape, tf.float32)
        width = tf.cast(tf.multiply(tf.div(shape[1], shape[0]), cfg.src_fixed_height, "imgWidth"), tf.int32)

        return tf.image.resize_images(img, [cfg.src_fixed_height, width],
                                      method=tf.image.ResizeMethod.BICUBIC)

    img = tf.cond(tf.not_equal(shape[0], cfg.src_fixed_height), lambda: f1(shape, img), lambda: img)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img

def readSrcImg(path, src):
    filenames = tf.string_join([path, src, ".jpg"], separator="")
    img = tf.cast(tf.image.decode_jpeg(tf.read_file(filenames), channels=1), tf.float32)
    return normalize_input_img(img)

def get_iterator(src_dataset,
                 tgt_dataset,
                 tgt_vocab_table,
                 tgt_sos_id,
                 tgt_eos_id,
                 num_threads=4,
                 output_buffer_size=120000):
  src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))
  src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size)

  path = tf.string_join([cfg.data_set, "/", cfg.imgFolderPath, "/"], separator="")
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (readSrcImg(path, src), tgt),
      num_threads=num_threads,output_buffer_size=output_buffer_size)

  # tf.string_split: Split elements of source based on delimiter into a SparseTensor.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src, tf.string_split([tgt]).values),
      num_threads=num_threads,output_buffer_size=output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_threads=num_threads,output_buffer_size=output_buffer_size)

  if cfg.src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:,:cfg.src_max_len, :], tgt),
        num_threads=num_threads,output_buffer_size=output_buffer_size)
  if cfg.tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:cfg.tgt_max_len]),
        num_threads=num_threads,output_buffer_size=output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_threads=num_threads,output_buffer_size=output_buffer_size)

  # Add in the word counts.  Subtract one from the target to avoid counting
  # the target_input <eos> tag (resp. target_output <sos> tag).
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.shape(src)[1], tf.size(tgt_in)),
      # src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_threads=num_threads,output_buffer_size=output_buffer_size)
  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        cfg.batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(tf.TensorShape([cfg.src_fixed_height, None, 1]),  # src
                       tf.TensorShape([None]),  # tgt_input
                       tf.TensorShape([None]),  # tgt_output
                       tf.TensorShape([]),  # src_len
                       tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(0.,  # src
                        tgt_eos_id,  # tgt_input
                        tgt_eos_id,  # tgt_output
                        0,
                        0))          # tgt_len -- unused
  if cfg.num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if cfg.src_max_len:
        bucket_width = (cfg.src_max_len + cfg.num_buckets - 1) // cfg.num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(cfg.num_buckets, bucket_id))
    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)
    batched_dataset = src_tgt_dataset.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=cfg.batch_size)
  else:
    batched_dataset = batching_func(src_tgt_dataset)

  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
      batched_iter.get_next())
  batchedInput = BatchedInput(
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
  return batched_iter.initializer, batchedInput
