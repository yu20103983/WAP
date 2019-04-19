# -*- coding: utf-8 -*-
# Tensorflow implementation WAP_Model for paper "
# Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition"
# at https://www.sciencedirect.com/science/article/pii/S0031320317302376
# author yuyufeng e-mail: yufeng_yu@sina.com
import tensorflow as tf
from config import cfg
import vocab_utils
import utils
import traceback
import time
from read_tf_records import *
import numpy as np
import shutil

from GRU_Att_Cov import GRU_Att_Cov
from FeatureCNN import FeatureCNN


def get_collections_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)


def get_local_time():
    timeStamp = int(time.time())
    timeArray = time.localtime(timeStamp)
    return time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)


class WAP():
    def __init__(self, is_training=True, checkPoint_path=None):
        self.graph = tf.Graph()
        self.is_training = is_training
        with self.graph.as_default():
            ano_data_set = os.path.join(cfg.data_set, cfg.ano_data_set)
            vocab_file = os.path.join(ano_data_set, cfg.tgt_vocab_file)
            vocab_size, vocab_file = vocab_utils.check_vocab(vocab_file, out_dir=cfg.out_dir, sos=cfg.sos, eos=cfg.eos,
                                                             unk=cfg.unk)

            self.tgt_vocab_table = vocab_utils.create_vocab_tables(vocab_file)
            self.reverse_tgt_vocab_table = vocab_utils.index_to_string_table_from_file(
                vocab_file, default_value=cfg.unk)

            self.tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(cfg.sos)), tf.int32)
            self.tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(cfg.eos)), tf.int32)

            if is_training:
                # train_src_dataset = tf.contrib.data.TextLineDataset(os.path.join(ano_data_set, cfg.train_src_dataset))
                # train_tgt_dataset = tf.contrib.data.TextLineDataset(os.path.join(ano_data_set, cfg.train_tgt_dataset))
                self.init_iter_train, self.iterator_train = get_iterator(cfg.train_tf_filename, self.tgt_vocab_table,
                                                                         self.tgt_sos_id, self.tgt_eos_id, augment=True)

                # vaild_src_dataset = tf.contrib.data.TextLineDataset(os.path.join(ano_data_set, cfg.vaild_src_dataset))
                # vaild_tgt_dataset = tf.contrib.data.TextLineDataset(os.path.join(ano_data_set, cfg.vaild_tgt_dataset))
                self.init_iter_vaild, self.iterator_vaild = get_iterator(cfg.vaild_tf_filename, self.tgt_vocab_table,
                                                                         self.tgt_sos_id, self.tgt_eos_id)

            else:
                self.source = tf.placeholder(tf.float32, (None, None), name='source')
                batch_source = tf.expand_dims(tf.expand_dims(self.source, axis=0), axis=-1)
                iterator_source = normalize_input_img(batch_source)
                self.source_sequence_length = tf.constant(tf.shape(iterator_source)[2], tf.int32)
                self.iterator = BatchedInput(source=iterator_source,
                                             target_input=None, target_output=None,
                                             source_sequence_length=self.source_sequence_length,
                                             target_sequence_length=None)

            self.featureCNN = FeatureCNN()
            self.gru_att_cov = GRU_Att_Cov(vocab_size) #词表size

            if is_training:
                if cfg.outer_batch_size:
                    outer_loss = 0
                    with tf.variable_scope('outer_batch_size') as scope:
                        for i in range(cfg.outer_batch_size):
                            if i > 0:
                                scope.reuse_variables()
                            self.cnn_out_train = self.featureCNN(self.iterator_train.source, True, False)
                            self.logits_train, _, self.attn_dists_train = self.gru_att_cov(self.cnn_out_train,
                                                                                           self.iterator_train, True,
                                                                                           self.tgt_sos_id)
                            outer_loss += self._loss(self.logits_train, self.iterator_train)

                    self.loss_train = outer_loss / cfg.outer_batch_size
                else:
                    self.cnn_out_train = self.featureCNN(self.iterator_train.source, True, False)
                    self.logits_train, _, self.attn_dists_train = self.gru_att_cov(self.cnn_out_train,
                                                                                   self.iterator_train, True,
                                                                                   self.tgt_sos_id)
                    self.loss_train = self._loss(self.logits_train, self.iterator_train)

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.exponential_decay(cfg.startLr, self.global_step, cfg.decay_steps,
                                                                cfg.decay_rate)
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
                self.train_op = optimizer.minimize(self.loss_train, global_step=self.global_step)

                self.cnn_out_vaild = self.featureCNN(self.iterator_vaild.source, True)
                self.logits_vaild, _, _ = self.gru_att_cov(self.cnn_out_vaild, self.iterator_vaild, True, self.tgt_sos_id)
                self.loss_vaild = self._loss(self.logits_vaild, self.iterator_vaild)

                self.cnn_out_vaild_infer = self.featureCNN(self.iterator_vaild.source, False)
                _, self.infer_indes_vaild, self.infer_attn_dists_vaild = self.gru_att_cov(self.cnn_out_vaild_infer,
                                                                                     self.iterator_vaild, False,
                                                                                     self.tgt_sos_id)
                self.infer_lookUpTgt_vaild = self.reverse_tgt_vocab_table.lookup(tf.to_int64(self.infer_indes_vaild))

                self.accuracy_vaild = self._acc(self.infer_indes_vaild, self.iterator_vaild.target_output)
                self.train_lookUpTgt_vaild = self.reverse_tgt_vocab_table.lookup(
                    tf.to_int64(self.iterator_vaild.target_output))

                self.train_summary, self.vaild_summary = self._summary()
            else:
                self.cnn_out = self.featureCNN(self.iterator.source, is_training)
                _, self.infer_indes, self.infer_attn_dists = self.gru_att_cov(self.cnn_out, self.iterator, False,
                                                                         self.tgt_sos_id)
                self.infer_lookUpTgt = self.reverse_tgt_vocab_table.lookup(tf.to_int64(self.infer_indes))

            self.init = [tf.global_variables_initializer(), tf.tables_initializer()]
            self.saver = tf.train.Saver()
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            if not is_training:
                self.sess.run(self.init)
                self.saver.restore(self.sess, checkPoint_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def _loss(self, logits, iterator):
        """Compute optimization loss."""
        target_output = iterator.target_output

        max_time = tf.shape(target_output)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            iterator.target_sequence_length, max_time, dtype=logits.dtype)
        return tf.reduce_mean(tf.div(tf.reduce_sum(
            crossent * target_weights, axis=1), tf.cast(iterator.target_sequence_length, tf.float32)))

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/learning_rate', self.learning_rate))
        train_summary.append(tf.summary.scalar('train/loss', self.loss_train))

        vaild_summary = []
        vaild_summary.append(tf.summary.scalar('vaild/loss', self.loss_vaild))
        vaild_summary.append(tf.summary.scalar('vaild/acc', self.accuracy_vaild))

        source_shape = tf.shape(self.iterator_train.source)
        source_h = source_shape[1]
        source_w = source_shape[2]

        infer_attn_dists_train_shape = tf.shape(self.attn_dists_train)
        batch_size = infer_attn_dists_train_shape[0]
        attn_times = infer_attn_dists_train_shape[1]
        cnn_strike_h = infer_attn_dists_train_shape[2]
        cnn_strike_w = infer_attn_dists_train_shape[3]

        infer_attn_dists_train_reshape = tf.reshape(self.attn_dists_train, (-1, cnn_strike_h, cnn_strike_w, 1))
        attn_dist_reSize = tf.image.resize_bicubic(infer_attn_dists_train_reshape, (source_h, source_w))
        attn_dist_reshape = tf.reshape(attn_dist_reSize, (batch_size, -1, source_w, 1))
        attn_dist_LitUp = attn_dist_reshape * 0.9 + 0.1 # 对比度和亮度调整
        attn_dist_LitUp = tf.clip_by_value(attn_dist_LitUp, 0, 1.)

        source_tile = tf.tile(self.iterator_train.source, [1, attn_times, 1, 1])
        source_mask_dist = tf.multiply(attn_dist_LitUp, source_tile)
        source_mask_dist = tf.clip_by_value(source_mask_dist, 0, 1.)
        train_summary.append(tf.summary.image('train/source_mask_dist', source_mask_dist, 8))

        return tf.summary.merge(train_summary), tf.summary.merge(vaild_summary)

    def _acc(self, infer_indes, target_output):
        infer_indes_slice = tf.slice(infer_indes, tf.zeros((tf.rank(infer_indes),), tf.int32),
                                     tf.shape(target_output))
        correct_prediction = tf.reduce_all(tf.equal(tf.to_int32(target_output), infer_indes_slice), axis=1)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, train_writer_path, log_f, restore_folder=None, restore_name=None):
        if not self.is_training: return
        sess = self.sess
        saver = self.saver
        if train_writer_path:
            train_writer = tf.summary.FileWriter(train_writer_path, sess.graph)
        time.clock()
        sess.run(self.init)
        if restore_folder:
            restore_path = os.path.join(restore_folder, restore_name)
            saver.restore(sess, restore_path)
            utils.print_out('load from %s' % restore_path, log_f)
            # with self.graph.as_default():
            #     gru_att_cov_variables = []
            #     gru_att_cov_variables.extend(get_collections_from_scope('tgt_embedding'))
            #     gru_att_cov_variables.extend(get_collections_from_scope('gru'))
            #     gru_att_cov_variables.extend(get_collections_from_scope('att'))
            #     gru_att_cov_variables.extend(get_collections_from_scope('coverage'))
            #     gru_att_cov_variables.extend(get_collections_from_scope('project'))
            #     init_gru_att_cov_variables = tf.variables_initializer(var_list=gru_att_cov_variables)
            # sess.run(init_gru_att_cov_variables)
            # utils.print_out('random init gru_att_cov variables', log_f)
        global_step = 0
        epoch = 0
        utils.print_hparams(cfg, f=log_f)
        sess.run(self.init_iter_train)  # 初始化训练输入
        utils.print_out('init_iter_train', log_f)
        sess.run(self.init_iter_vaild)  # 初始化验证输入
        utils.print_out('init_iter_vaild', log_f)
        learning_rate = 0
        loss = 0
        try:
            utils.print_out("start at %s" % get_local_time(), log_f)
            while epoch < cfg.epochs:
                epoch += 1
                i_steps = 0

                while i_steps < cfg.each_steps:
                    try:
                        _, loss, global_step, learning_rate, summary = \
                            sess.run([self.train_op, self.loss_train, self.global_step,
                                      self.learning_rate, self.train_summary])
                        i_steps += 1
                    except tf.errors.OutOfRangeError:
                        sess.run(self.init_iter_train)
                    if global_step % cfg.print_frq == 0:
                        utils.print_out('epoch %d, step %d, gloSp %d, lr %.4f, loss %.4f'
                                        % (epoch, i_steps, global_step, learning_rate, loss), log_f)
                    if global_step % cfg.summary_frq == 0:
                        summary = sess.run(self.train_summary)
                        if train_writer_path: train_writer.add_summary(summary, global_step=global_step)

                if epoch % cfg.val_frq == 0:

                    val_count = 0
                    val_loss = 0
                    val_acc = 0
                    val_edit_dist = 0
                    true_sample_words = ['']
                    pred_sample_words = ['']
                    i_loss = 0
                    i_acc = 0
                    while val_count < cfg.val_steps:
                        try:
                            i_loss, i_acc, true_sample_words, pred_sample_words, summary = \
                                sess.run([self.loss_vaild, self.accuracy_vaild, self.train_lookUpTgt_vaild,
                                          self.infer_lookUpTgt_vaild, self.vaild_summary])
                            val_count += 1
                        except tf.errors.OutOfRangeError:
                            sess.run(self.init_iter_vaild)

                        val_loss += i_loss
                        val_acc += i_acc
                        c_val_edit_dist = []
                        for t, p in zip(true_sample_words, pred_sample_words):
                            edit_dist = utils.normal_leven(t, p)
                            c_val_edit_dist.append(edit_dist)
                        c_val_edit_dist = sum(c_val_edit_dist) / float(len(c_val_edit_dist))
                        val_edit_dist += c_val_edit_dist

                    val_acc /= val_count
                    val_loss /= val_count
                    val_edit_dist /= val_count

                    timeStamp = int(time.time())
                    timeArray = time.localtime(timeStamp)
                    styleTime = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)

                    utils.print_out('%s ### val loss %.4f, acc %.4f, edit_dist %.4f'
                                    % (styleTime, val_loss, val_acc, val_edit_dist), log_f)
                    if train_writer_path: train_writer.add_summary(summary, global_step=global_step)
                    test_show_size = min(cfg.test_show_size, len(true_sample_words))
                    for i in range(test_show_size):
                        str_tr = ''.join(true_sample_words[i])
                        str_pd = ''.join(pred_sample_words[i])
                        utils.print_out("   ## true: %s" % (str_tr), log_f)
                        utils.print_out("      pred: %s" % (str_pd), log_f)
                if epoch % cfg.save_frq == 0 and train_writer_path:
                    checkPoint_path = os.path.join(train_writer_path, "checkPoint.model")
                    saver.save(sess, checkPoint_path, global_step=global_step)
                    utils.print_out(
                        "   global step %d, check point save to %s-%d" % (
                            global_step, checkPoint_path, global_step), log_f)

        except Exception as e:
            utils.print_out(
                "!!!!  Interrupt ## end training, global step %d" % (
                    global_step), log_f)
            if len(e.args) > 0:
                utils.print_out("An error occurred. {}".format(e.args[-1]), log_f)
            traceback.print_exc()

        finally:
            if train_writer_path:
                checkPoint_path = os.path.join(train_writer_path, "end_checkPoint.model")
                saver.save(sess, checkPoint_path, global_step=global_step)
                utils.print_out(
                    "   end training, global step %d, check point save to %s-%d" % (
                        global_step, checkPoint_path, global_step), log_f)
            utils.print_out("end at %s" % get_local_time(), log_f)
            return epoch

    def predict(self, img):
        if self.is_training: return
        assert np.rank(img) == 2
        tgt = self.sess.run([self.infer_lookUpTgt], feed_dict={self.source: img})
        return tgt


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    timeStamp = get_local_time()
    out_dir = os.path.join(cfg.out_dir, "%s" % timeStamp)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, "log")
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_file=%s" % log_file, log_f)
    wap = WAP(True)
    start = time.time()
    epoch = wap.train(out_dir, log_f, cfg.load_preTrain_model_folder, cfg.load_preTrain_model_name)
    end = time.time()
    time_consum = (end - start) / 3600
    if cfg.debug and time_consum < 0.5:
        print('log folder %s removed' % out_dir)
        shutil.rmtree(out_dir)
    else:
        shutil.move(out_dir, os.path.join(cfg.out_dir, "%03d_%04d_%s" % (time_consum * 10, epoch, timeStamp)))
