# -*- coding: utf-8 -*-
import tensorflow as tf
from config import cfg
from tensorflow.python.training.moving_averages import assign_moving_average


def tfVariable(dtype, shape, name, trainable=True):
    return tf.Variable(tf.truncated_normal(dtype=dtype, shape=shape, mean=0, stddev=0.01), name=name,
                       trainable=trainable)


def tfVariable_ones(dtype, shape, name, trainable=True):
    return tf.Variable(tf.ones(dtype=dtype, shape=shape), name=name, trainable=trainable)


def tfVariable_zeros(dtype, shape, name, trainable=True):
    return tf.Variable(tf.zeros(dtype=dtype, shape=shape), name=name, trainable=trainable)


class FeatureCNN():
    def __init__(self):
        with tf.variable_scope('featureCNN'):
            # 第一组
            self.conv1_1_w = tfVariable(tf.float32, (5, 5, 1, 16), 'conv1_1_w')
            self.scale1_1 = tfVariable_ones(tf.float32, 16, 'scale1_1')
            self.shift1_1 = tfVariable_zeros(tf.float32, 16, 'shift1_1')
            self.var1_1 = tfVariable_ones(tf.float32, 16, 'var1_1', False)
            self.mean1_1 = tfVariable_zeros(tf.float32, 16, 'mean1_1', False)
            conv1_1 = (self.conv1_1_w, self.scale1_1, self.shift1_1, self.var1_1, self.mean1_1)
            # self.conv1_2_w = tfVariable(tf.float32, (3, 3, 32, 32), 'conv1_2_w')
            # self.scale1_2 = tfVariable_ones(tf.float32, 32, 'scale1_2')
            # self.shift1_2 = tfVariable_zeros(tf.float32, 32, 'shift1_2')
            # self.var1_2 = tfVariable_ones(tf.float32, 32, 'var1_2', False)
            # self.mean1_2 = tfVariable_zeros(tf.float32, 32, 'mean1_2', False)
            # conv1_2 = (self.conv1_2_w, self.scale1_2, self.shift1_2)
            # self.conv1_3_w = tfVariable(tf.float32, (3, 3, 32, 32), 'conv1_3_w')
            # self.scale1_3 = tfVariable_ones(tf.float32, 32, 'scale1_3')
            # self.shift1_3 = tfVariable_zeros(tf.float32, 32, 'shift1_3')
            # conv1_3 = (self.conv1_3_w, self.scale1_3, self.shift1_3)
            # self.conv1_4_w = tfVariable(tf.float32, (3, 3, 32, 32), 'conv1_4_w')
            # self.scale1_4 = tfVariable_ones(tf.float32, 32, 'scale1_4')
            # self.shift1_4 = tfVariable_zeros(tf.float32, 32, 'shift1_4')
            # conv1_4 = (self.conv1_4_w, self.scale1_4, self.shift1_4)
            # self.conv1 = (conv1_1, conv1_2, conv1_3, conv1_4)
            # self.conv1 = (conv1_1, conv1_2)
            self.conv1 = (conv1_1,)

            # 第二组
            self.conv2_1_w = tfVariable(tf.float32, (3, 3, 16, 16), 'conv2_1_w')
            self.scale2_1 = tfVariable_ones(tf.float32, 16, 'scale2_1')
            self.shift2_1 = tfVariable_zeros(tf.float32, 16, 'shift2_1')
            self.var2_1 = tfVariable_ones(tf.float32, 16, 'var2_1', False)
            self.mean2_1 = tfVariable_zeros(tf.float32, 16, 'mean2_1', False)
            conv2_1 = (self.conv2_1_w, self.scale2_1, self.shift2_1, self.var2_1, self.mean2_1)
            # self.conv2_2_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv2_2_w')
            # self.scale2_2 = tfVariable_ones(tf.float32, 64, 'scale2_2')
            # self.shift2_2 = tfVariable_zeros(tf.float32, 64, 'shift2_2')
            # conv2_2 = (self.conv2_2_w, self.scale2_2, self.shift2_2)
            # self.conv2_3_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv2_3_w')
            # self.scale2_3 = tfVariable_ones(tf.float32, 64, 'scale2_3')
            # self.shift2_3 = tfVariable_zeros(tf.float32, 64, 'shift2_3')
            # conv2_3 = (self.conv2_3_w, self.scale2_3, self.shift2_3)
            # self.conv2_4_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv2_4_w')
            # self.scale2_4 = tfVariable_ones(tf.float32, 64, 'scale2_4')
            # self.shift2_4 = tfVariable_zeros(tf.float32, 64, 'shift2_4')
            # conv2_4 = (self.conv2_4_w, self.scale2_4, self.shift2_4)
            # self.conv2 = (conv2_1, conv2_2, conv2_3, conv2_4)
            # self.conv2 = (conv2_1, conv2_2)
            self.conv2 = (conv2_1,)

            # 第三组
            self.conv3_1_w = tfVariable(tf.float32, (3, 3, 16, 32), 'conv3_1_w')
            self.scale3_1 = tfVariable_ones(tf.float32, 32, 'scale3_1')
            self.shift3_1 = tfVariable_zeros(tf.float32, 32, 'shift3_1')
            self.var3_1 = tfVariable_ones(tf.float32, 32, 'var3_1', False)
            self.mean3_1 = tfVariable_zeros(tf.float32, 32, 'mean3_1', False)
            conv3_1 = (self.conv3_1_w, self.scale3_1, self.shift3_1, self.var3_1, self.mean3_1)
            # self.conv3_2_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv3_2_w')
            # self.scale3_2 = tfVariable_ones(tf.float32, 64, 'scale3_2')
            # self.shift3_2 = tfVariable_zeros(tf.float32, 64, 'shift3_2')
            # conv3_2 = (self.conv3_2_w, self.scale3_2, self.shift3_2)
            # self.conv3_3_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv3_3_w')
            # self.scale3_3 = tfVariable_ones(tf.float32, 64, 'scale3_3')
            # self.shift3_3 = tfVariable_zeros(tf.float32, 64, 'shift3_3')
            # conv3_3 = (self.conv3_3_w, self.scale3_3, self.shift3_3)
            # self.conv3_4_w = tfVariable(tf.float32, (3, 3, 64, 64), 'conv3_4_w')
            # self.scale3_4 = tfVariable_ones(tf.float32, 64, 'scale3_4')
            # self.shift3_4 = tfVariable_zeros(tf.float32, 64, 'shift3_4')
            # conv3_4 = (self.conv3_4_w, self.scale3_4, self.shift3_4)
            # self.conv3 = (conv3_1, conv3_2, conv3_3, conv3_4)
            # self.conv3 = (conv3_1, conv3_2)
            self.conv3 = (conv3_1,)

            # 第四组
            self.conv4_1_w = tfVariable(tf.float32, (3, 3, 32, cfg.rnn_input_dimensions), 'conv4_1_w')
            self.scale4_1 = tfVariable_ones(tf.float32, cfg.rnn_input_dimensions, 'scale4_1')
            self.shift4_1 = tfVariable_zeros(tf.float32, cfg.rnn_input_dimensions, 'shift4_1')
            self.var4_1 = tfVariable_ones(tf.float32, cfg.rnn_input_dimensions, 'var4_1', False)
            self.mean4_1 = tfVariable_zeros(tf.float32, cfg.rnn_input_dimensions, 'mean4_1', False)
            conv4_1 = (self.conv4_1_w, self.scale4_1, self.shift4_1, self.var4_1, self.mean4_1)
            # self.conv4_2_w = tfVariable(tf.float32, (3, 3, cfg.rnn_input_dimensions, cfg.rnn_input_dimensions), 'conv4_2_w')
            # self.scale4_2 = tfVariable_ones(tf.float32, cfg.rnn_input_dimensions, 'scale4_2')
            # self.shift4_2 = tfVariable_zeros(tf.float32, cfg.rnn_input_dimensions, 'shift4_2')
            # conv4_2 = (self.conv4_2_w, self.scale4_2, self.shift4_2)
            # self.conv4_3_w = tfVariable(tf.float32, (3, 3, cfg.rnn_input_dimensions, cfg.rnn_input_dimensions), 'conv4_3_w')
            # self.scale4_3 = tfVariable_ones(tf.float32, cfg.rnn_input_dimensions, 'scale4_3')
            # self.shift4_3 = tfVariable_zeros(tf.float32, cfg.rnn_input_dimensions, 'shift4_3')
            # conv4_3 = (self.conv4_3_w, self.scale4_3, self.shift4_3)
            # self.conv4_4_w = tfVariable(tf.float32, (3, 3, cfg.rnn_input_dimensions, cfg.rnn_input_dimensions), 'conv4_4_w')
            # self.scale4_4 = tfVariable_ones(tf.float32, cfg.rnn_input_dimensions, 'scale4_4')
            # self.shift4_4 = tfVariable_zeros(tf.float32, cfg.rnn_input_dimensions, 'shift4_4')
            # conv4_4 = (self.conv4_4_w, self.scale4_4, self.shift4_4)
            # self.conv4 = (conv4_1, conv4_2, conv4_3, conv4_4)
            # self.conv4 = (conv4_1, conv4_2)
            self.conv4 = ((conv4_1),)

    def __call__(self, input, is_training=True, bn_mv = True):
        def Batch_Norn(s_input, scale, shift, moving_variance, moving_mean, axis=[0, 1, 2], eps=1e-05, decay=0.9,
                       name=None):
            def mean_var_with_update():
                means, variances = tf.nn.moments(s_input, axes=axis, name='moments')
                with tf.variable_scope('ass_m_a_%s' % name if name else 'ass_m_a', reuse=bn_mv):
                    with tf.control_dependencies([assign_moving_average(moving_mean, means, decay, zero_debias=False),
                                                  assign_moving_average(moving_variance, variances, decay, zero_debias=False)]):
                        return tf.identity(means), tf.identity(variances)

            if bn_mv:
                mean = moving_mean
                var = moving_variance
            else:
                mean, var = mean_var_with_update()

            return tf.nn.batch_normalization(s_input, mean, var, shift, scale, eps, name=name)

        def CNNBlock(s_input, W, pool_size=[2, 2], bn=True, dropout=None, name=None):
            out = [s_input]
            for i, w_s_s in enumerate(W):
                if bn:
                    w, scale, shift, var, mean = w_s_s
                else:
                    w, b = w_s_s[0]
                conv = tf.nn.conv2d(out[-1], w, [1, 1, 1, 1], "SAME", name=(name + "_%d_c2d" % i if name else name))
                out.append(conv)

                conv = tf.nn.relu(conv, name=(name + "_%d_relu" % i if name else name))
                out.append(conv)
                if bn:
                    conv = Batch_Norn(conv, scale, shift, var, mean, name=(name + "_%d_BN" % i if name else name))
                    # conv = tf.nn.bias_add(conv, shift, name=(name + "_%d_Bias" % i if name else name))
                else:
                    conv = tf.nn.bias_add(conv, b, name=(name + "_%d_Bias" % i if name else name))
                out.append(conv)

                if is_training and dropout:
                    conv = tf.nn.dropout(conv, keep_prob=dropout, name=(name + "_%d_dp" % i if name else name))
                    out.append(conv)
            pool = tf.nn.max_pool(out[-1], [1, pool_size[0], pool_size[1], 1], [1, pool_size[0], pool_size[1], 1],
                                  "SAME", name=(name + "_pool" if name else name))
            out.append(pool)
            return out

        def FCBlock(s_input, W, bn=True, act=True, dropout=None, name=None):
            out = [s_input]
            for i, w_s_s in enumerate(W):
                if bn:
                    w, scale, shift, var, mean = w_s_s
                else:
                    w, b = w_s_s

                fc = tf.matmul(out[-1], w, name=(name + "_%d_fc" % i if name else name))
                out.append(fc)
                if act: fc = tf.nn.relu(fc, name=(name + "_%d_relu" % i if name else name))
                out.append(fc)
                if bn:
                    fc = Batch_Norn(fc, scale, shift, var, mean, axis=[0], name=(name + "_%d_BN" % i if name else name))
                else:
                    fc = tf.nn.bias_add(fc, b, name=(name + "_%d_Bias" % i if name else name))
                out.append(fc)

                if is_training and dropout:
                    fc = tf.nn.dropout(fc, keep_prob=dropout, name=(name + "_%d_dp" % i if name else name))
                    out.append(fc)
            return out

        cnn_1 = CNNBlock(input, self.conv1, pool_size=[3, 3], name='b1')[-1]
        cnn_2 = CNNBlock(cnn_1, self.conv2, pool_size=[2, 2], name='b2')[-1]
        cnn_3 = CNNBlock(cnn_2, self.conv3, pool_size=[2, 2], name='b3')[-1]
        cnn_4 = CNNBlock(cnn_3, self.conv4, pool_size=[2, 2], dropout=cfg.keep_prob, name='b4')[-1]
        return cnn_4
