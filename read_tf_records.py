#coding=utf-8

import tensorflow as tf
import os
import collections
import numpy as np
import Augment
from config import cfg
import vocab_utils


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
    img = tf.cast(img, tf.float32) * (1. / 255)

    return img

def get_iterator(tf_filename, tgt_vocab_table,
                 tgt_sos_id,
                 tgt_eos_id, repeat=None,
                 num_threads=4,
                 output_buffer_size=120000, augment=False):
    dataset = tf.contrib.data.TFRecordDataset(os.path.join(cfg.data_set, cfg.ano_data_set, tf_filename))

    def parser_tfrecord(record):
        parsed = tf.parse_single_example(record,
                                         features={
                                             'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                                             'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                                             'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
                                             'label/value': tf.VarLenFeature(tf.string),
                                         })

        # img = tf.cast(tf.image.decode_jpeg(parsed['image/encoded'], channels=1), tf.float32) # 保存的时候先按jpeg编码
        img = tf.decode_raw(parsed['image/encoded'], tf.float32) #直接采用bytes编码
        height = tf.cast(parsed['image/height'], tf.int32)
        width = tf.cast(parsed['image/width'], tf.int32)
        img = tf.reshape(img, (height, width, 1))
        img = tf.clip_by_value((img + 0.5), 0, 1.)* 255.
        img = normalize_input_img(img)
        if augment: img = Augment.augment(img)
        label = tf.sparse_tensor_to_dense(parsed['label/value'], default_value='')

        return img, label

    dataset = dataset.map(parser_tfrecord, num_threads=num_threads, output_buffer_size=output_buffer_size)
    dataset = dataset.map(
        lambda src, tgt: (src, tf.string_split(tgt).values),
        num_threads=num_threads, output_buffer_size=output_buffer_size)

    dataset = dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    dataset = dataset.map(
        lambda src, tgt: (src, tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    if cfg.src_max_len:
        # dataset = dataset.map(
        #     lambda src, tgt: (src[:, :cfg.src_max_len, :], tgt),
        #     num_threads=num_threads, output_buffer_size=output_buffer_size)
        dataset = dataset.filter(
            lambda src, tgt: tf.shape(src)[1] <= cfg.src_max_len)
    if cfg.tgt_max_len:
        dataset = dataset.filter(
            lambda src, tgt: tf.size(tgt) <= cfg.tgt_max_len)
        # dataset = dataset.map(
        #     lambda src, tgt: (src, tgt[:cfg.tgt_max_len]),
        #     num_threads=num_threads, output_buffer_size=output_buffer_size)

    # 对一些相似字符进行替换
    # def repalce_some_label(labels):
    #     def replace_int_label(a, rep, case, *o_case):
    #         condition = tf.equal(a, case)
    #         for i_case in o_case:
    #             condition = tf.logical_or(condition, tf.equal(a, i_case))
    #         case_true = tf.multiply(tf.ones_like(a, tf.int32), rep)
    #         case_false = a
    #         a_m = tf.where(condition, case_true, case_false)
    #         return a_m
    #
    #     labels = replace_int_label(labels, 47, 21, 61)  # C,c替换为括号
    #     labels = replace_int_label(labels, 6, 29, 72)  # O,o替换为0
    #     labels = replace_int_label(labels, 69, 26)  # L替换为l
    #     labels = replace_int_label(labels, 79, 34)  # V替换为v
    #     labels = replace_int_label(labels, 80, 35, 54)  # X，乘以替换为x
    #     labels = replace_int_label(labels, 47, 40)  # 角替换为小于
    #     labels = replace_int_label(labels, 76, 32)  # S替换为s
    #     return labels
    #
    # dataset = dataset.map(lambda src, tgt: (src, repalce_some_label(tgt)))

    dataset = dataset.map(
        lambda src, tgt: (src, tf.concat(([tgt_sos_id], tgt), 0), tf.concat((tgt, [tgt_eos_id]), 0)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    dataset = dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.shape(src)[1], tf.size(tgt_in)),
       num_threads=num_threads, output_buffer_size=output_buffer_size)

    def batching_func(x):
        return x.padded_batch(
            cfg.batch_size,
            padded_shapes=(tf.TensorShape([cfg.src_fixed_height, None, 1]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            padding_values=(0.,  # src
                            tgt_eos_id,  # tgt_input
                            tgt_eos_id,  # tgt_output
                            0,
                            0))  # tgt_len -- unused

    if cfg.num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            if cfg.src_max_len:
                bucket_width = (cfg.src_max_len + cfg.num_buckets - 1) // cfg.num_buckets
            else:
                bucket_width = cfg.default_bucket_width
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(cfg.num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=cfg.batch_size)
    else:
        batched_dataset = batching_func(dataset)
    batched_dataset = batched_dataset.shuffle(buffer_size=6000)
    if repeat: batched_dataset = batched_dataset.repeat(repeat)
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


def dense2sparse(dense):
    indices, values = zip(*[([i, j], val)
                            for i, row in enumerate(dense) for j, val in enumerate(row)])
    max_len = max([len(row) for row in dense])
    shape = [len(dense), max_len]
    sparse = (np.array(indices), np.array(values), np.array(shape))
    return sparse


def dense2sparse_second(sequences):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    sparse = (indices, values, shape)
    return sparse


def main(_):
    ano_data_set = os.path.join(cfg.data_set, cfg.ano_data_set)
    vocab_file = os.path.join(ano_data_set, cfg.tgt_vocab_file)
    
    with tf.Graph().as_default():
        vocab_size, vocab_file = vocab_utils.check_vocab(vocab_file, out_dir=cfg.out_dir, sos=cfg.sos, eos=cfg.eos,
                                                         unk=cfg.unk)

        tgt_vocab_table = vocab_utils.create_vocab_tables(vocab_file)
        reverse_tgt_vocab_table = vocab_utils.index_to_string_table_from_file(
            vocab_file, default_value=cfg.unk)

        tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(cfg.sos)), tf.int32)
        tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(cfg.eos)), tf.int32)
        iter, batch_input = get_iterator(cfg.vaild_tf_filename, tgt_vocab_table, tgt_sos_id,tgt_eos_id)
        lookUpTgt = reverse_tgt_vocab_table.lookup(tf.to_int64(batch_input.target_output))
        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sess.run(iter)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while True:
                try:
                    while not coord.should_stop():
                        src, tgt_output, src_seq_len, tgt_seq_len = \
                            sess.run([batch_input.source, lookUpTgt, batch_input.source_sequence_length, batch_input.target_sequence_length])
                        if np.isnan(np.max(src)) or np.isnan(np.min(src)):
                            print('get a nan')
                            exit(1)
                        if np.any(np.less(src, 0.)):
                            print('get a fushu')
                            exit(1)
                        print('run one')
                        step += 1
                except tf.errors.OutOfRangeError:
                    print('check finished')
                    exit(1)
                    sess.run(iter)
        except KeyboardInterrupt:
            print('interrupt')
        finally:
                coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tf.app.run()
