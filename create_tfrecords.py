#coding=utf-8
__author__ = 'Administrator'
import os
from config import cfg
import tensorflow as tf
import sys
import numpy as np
from skimage import io


def read_txt(file_name):
    with open(file_name, 'r', encoding='utf-8') as f_i:
        return f_i.readlines()

def int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

def bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

def create_tfrecords(tf_filename, src_dataset, tgt_dataset):

    img_names = read_txt(os.path.join(cfg.data_set, cfg.ano_data_set, src_dataset))
    labels = read_txt(os.path.join(cfg.data_set, cfg.ano_data_set, tgt_dataset))


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

    def readSrcImg(img_folder):
        src_holder = tf.placeholder(tf.string, ())
        filenames = tf.string_join([img_folder, '/', src_holder, ".jpg"], separator="")
        img = tf.cast(tf.image.decode_jpeg(tf.read_file(filenames), channels=1), tf.float32)
        norm_img = tf.reshape(normalize_input_img(img), (cfg.src_fixed_height, -1))
        return src_holder, norm_img

    img_folder = os.path.join(cfg.data_set, cfg.imgFolderName)
    src_holder, norm_img = readSrcImg(img_folder)
    with tf.Session() as sess:
        vocb = read_txt(os.path.join(cfg.data_set, cfg.ano_data_set, cfg.tgt_vocab_file))
        vocb = [ivocb.strip() for ivocb in vocb]
        with tf.python_io.TFRecordWriter(
                os.path.join(cfg.data_set, cfg.ano_data_set, tf_filename)) as tfrecord_writer:
            for i, img_name, label in zip(range(len(img_names)), img_names, labels):
                try:
                    image_data = sess.run(norm_img, feed_dict={src_holder: img_name.strip()})
                    label = label.strip()
                    if len(label) == 0:
                        recg_ind = []
                    else:
                        recg_ind = [vocb.index(word) for word in label.split(' ')]

                    label = bytes(label, encoding="utf8")
                    height, width = image_data.shape
                    image_data = image_data.tobytes()
                    example = tf.train.Example(
                        # 属性名称到取值的字典
                        features=tf.train.Features(feature={"image/encoded": bytes_feature(image_data),
                                                            'image/height': int64_feature(height),
                                                            'image/width': int64_feature(width),
                                                            "label/value": bytes_feature(label),
                                                            "label/ind": tf.train.Feature(
                                                                int64_list=tf.train.Int64List(value=recg_ind))}))

                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('%d of %d : %s' % (i, len(img_names), img_name))
                    sys.stdout.flush()
                except Exception as e:
                    print("Error: ", e)


    print('\nFinished converting the dataset!')

def mod_vocb_tfrecords(tf_filename, tf_filename_2, tgt_dataset):
    def read_tfrecord():
        dataset = tf.contrib.data.TFRecordDataset(os.path.join(cfg.data_set, cfg.ano_data_set, tf_filename))
        def parser_tfrecord(record):
            parsed = tf.parse_single_example(record,
                                             features={
                                                 'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                                                 'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                                                 'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
                                                 'label/value': tf.VarLenFeature(tf.string),
                                                 'label/ind': tf.VarLenFeature(tf.int64),
                                             })

            img = parsed['image/encoded']  # 直接采用bytes编码
            height = parsed['image/height']
            width = parsed['image/width']

            return img, height, width
        dataset = dataset.map(parser_tfrecord, num_threads=4, output_buffer_size=6000)
        dataset = dataset.batch(128)
        batched_iter = dataset.make_initializable_iterator()
        img, height, width = batched_iter.get_next()
        return batched_iter.initializer, (img, height, width)

    batched_iter, data = read_tfrecord()
    img_op, height_op, width_op = data
    sess = tf.Session()
    sess.run([tf.global_variables_initializer()])
    sess.run(batched_iter)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        vocb = read_txt(os.path.join(cfg.data_set, cfg.ano_data_set, cfg.tgt_vocab_file))
        vocb = [ivocb.strip() for ivocb in vocb]
        labels = read_txt(os.path.join(cfg.data_set, cfg.ano_data_set, tgt_dataset))

        with tf.python_io.TFRecordWriter(
                os.path.join(cfg.data_set, cfg.ano_data_set, tf_filename_2)) as tfrecord_writer:
            count = 0
            while not coord.should_stop():
                img, height, width = sess.run([img_op, height_op, width_op])
                for i_img, i_height, i_width in zip(img, height, width):
                    save_img =sess.run(tf.clip_by_value((tf.reshape(tf.decode_raw(tf.constant(i_img, tf.string), tf.float32), (i_height, i_width)) + 0.5), 0, 1.) * 255.)
                    io.imsave(os.path.join(cfg.debug_dir, '%d.jpg'%count), save_img)
                    label = labels[count].strip()
                    if len(label) == 0:
                        recg_ind = []
                    else:
                        recg_ind = [vocb.index(word) for word in label.split(' ')]
                    try:
                        example = tf.train.Example(
                            # 属性名称到取值的字典
                            features=tf.train.Features(feature={"image/encoded": bytes_feature(i_img),
                                                                'image/height': int64_feature(height),
                                                                'image/width': int64_feature(width),
                                                                "label/value": bytes_feature(label),
                                                                "label/ind": tf.train.Feature(
                                                                int64_list=tf.train.Int64List(value=recg_ind))}))

                        tfrecord_writer.write(example.SerializeToString())
                        sys.stdout.write('%d of %d' % (count, len(labels)))
                        sys.stdout.flush()
                        count += 1
                    except Exception as e:
                        print("Error: ", e)

    except tf.errors.OutOfRangeError:
        print('finished')
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    create_tfrecords(cfg.train_tf_filename, cfg.train_src_dataset, cfg.train_tgt_dataset)
    create_tfrecords(cfg.vaild_tf_filename, cfg.vaild_src_dataset, cfg.vaild_tgt_dataset)
    # mod_vocb_tfrecords(cfg.train_tf_filename, cfg.train_tf_filename + 'mod', cfg.train_tgt_dataset)
    # mod_vocb_tfrecords(cfg.vaild_tf_filename, cfg.vaild_tf_filename + 'mod', cfg.vaild_tgt_dataset)