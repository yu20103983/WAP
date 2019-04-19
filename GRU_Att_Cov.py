# -*- coding: utf-8 -*-
import tensorflow as tf
from config import cfg

def tfVariable(dtype, shape, name):
    return tf.Variable(tf.truncated_normal(dtype=dtype, shape=shape, mean=0, stddev=0.01), name=name)
class GRU_Att_Cov():
    def __init__(self, tgt_table_size):
        self.input_dimensions = cfg.rnn_input_dimensions
        self.hidden_size = cfg.rnn_hidden_size
        self.attention_size = cfg.attention_size
        self.coverage_size = cfg.coverage_size
        self.embedding_size = cfg.tgt_embedding_size
        self.project_size = tgt_table_size
        with tf.variable_scope('tgt_embedding'):
            self.embedding = tfVariable(tf.float32, (self.project_size, self.embedding_size), 'embedding')
        with tf.variable_scope('gru') as scope:
            # Weights for input vectors of shape (input_dimensions, hidden_size)
            self.wr = tfVariable(tf.float32, (self.input_dimensions, self.hidden_size), 'wr')
            self.wz = tfVariable(tf.float32, (self.input_dimensions, self.hidden_size), 'wz')
            self.wh = tfVariable(tf.float32, (self.input_dimensions, self.hidden_size), 'wh')
            # Weights for hidden vectors of shape (hidden_size, hidden_size)
            self.ur = tfVariable(tf.float32, (self.hidden_size, self.hidden_size), 'ur')
            self.uz = tfVariable(tf.float32, (self.hidden_size, self.hidden_size), 'uz')
            self.uh = tfVariable(tf.float32, (self.hidden_size, self.hidden_size), 'uh')
            # Weights for embedding vectors of shape (hidden_size, hidden_size)
            self.yr = tfVariable(tf.float32, (self.embedding_size, self.hidden_size), 'yr')
            self.yz = tfVariable(tf.float32, (self.embedding_size, self.hidden_size), 'yz')
            self.yh = tfVariable(tf.float32, (self.embedding_size, self.hidden_size), 'yh')
            # Biases for hidden vectors of shape (hidden_size,)
            self.br = tfVariable(tf.float32, (self.hidden_size,), 'br')
            self.bz = tfVariable(tf.float32, (self.hidden_size,), 'bz')
            self.bh = tfVariable(tf.float32, (self.hidden_size,), 'bh')
        with tf.variable_scope('att'):
            # Weights for attention mechanism hidden vectors of shape (hidden_size, attention_size)
            self.wa = tfVariable(tf.float32, (self.hidden_size, self.attention_size), 'wa')
            # Weights for attention mechanism hidden input of shape (input_dimensions, attention_size)
            self.ua = tfVariable(tf.float32, (self.input_dimensions, self.attention_size), 'ua')
            # Weights for attention mechanism of shape (attention_size)
            self.va = tfVariable(tf.float32, (self.attention_size,), 'va')
        with tf.variable_scope('coverage'):
            conv1_w = tfVariable(tf.float32, (11, 11, 1, self.coverage_size), 'conv1_w')
            conv1_b = tfVariable(tf.float32, (self.coverage_size,), 'conv1_b')
            def ConvCoverage(input):
                conv1 = tf.nn.conv2d(input, conv1_w, [1, 1, 1, 1], "SAME", name='conv1')
                conv1_addb = tf.nn.bias_add(conv1, conv1_b)
                conv1_act = tf.nn.relu(conv1_addb)
                return conv1_act
            # Conv weights for Coverage mechanism of shape ()
            self.convCoverage = ConvCoverage
            self.uc = tfVariable(tf.float32, (self.coverage_size, self.attention_size), 'uc')
        with tf.variable_scope('project'):
            self.ph = tfVariable(tf.float32, (self.hidden_size, self.embedding_size), 'ph')
            self.pc = tfVariable(tf.float32, (self.input_dimensions, self.embedding_size), 'pc')
            self.po = tfVariable(tf.float32, (self.embedding_size, self.project_size), 'po')
            self.pb = tfVariable(tf.float32, (self.project_size,), 'pb')

    def __call__(self, cnn_out, iterator, is_training=True, tgt_sos_id = None):
        source = cnn_out
        h = tf.shape(source, name='shape_h')[1]
        w = tf.shape(source, name='shape_w')[2]
        batch_size = tf.shape(source, name='shape_batch_size')[0]
        source_reshape = tf.reshape(source, (batch_size, -1, self.input_dimensions), name='reshape_1')
        source_dist_major = tf.transpose(source_reshape, (1, 0 ,2), name='transpose_1')

        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0 and initial cover state cover_0
        h_0 = tf.matmul(source_dist_major[0], tf.zeros((self.input_dimensions, self.hidden_size), tf.float32), name='matmul_1')
        cover_0 = tf.zeros((batch_size, h, w, 1), tf.float32)
        attn_dist_0 = tf.zeros((batch_size, h, w, 1), tf.float32)
        inds_t_1 = tf.fill((batch_size,), tgt_sos_id)
        logits_0 = tf.zeros((batch_size, self.project_size), tf.float32)

        t_0_state = (h_0, cover_0, logits_0, inds_t_1, attn_dist_0)

        padding_mask = tf.sequence_mask(tf.cast(tf.ceil(tf.divide(iterator.source_sequence_length, cfg.cnn_strike)), tf.int32), w, name='sequence_mask')
        padding_mask_tile = tf.tile(padding_mask, [1, h], name='tile_1')
        padding_mask_tile = tf.cast(padding_mask_tile, tf.float32)

        def masked_attention(e):
          attn_dist = tf.nn.softmax(e, name='softmax_1')
          attn_dist *= padding_mask_tile # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1, name='reduce_sum_1')
          return attn_dist / tf.reshape(masked_sums, [-1, 1], name='reshape_2') # re-normalize

        def forward_pass(t_1_state, t_input):
            h_t_1, cover_t_1, logits_t_1, inds_t_1, _ = t_1_state #
            # WAP (13)
            F = self.convCoverage(cover_t_1)
            F_flatten = tf.reshape(F, (batch_size, -1, self.coverage_size), name='reshape_3')
            F_flatten_dist_major = tf.transpose(F_flatten, (1, 0, 2), name='transpose_3')
            # (14)

            def att_Clcu(a_t_1_state, a_t_input):
                slice_source, slice_F = a_t_input
                ha = tf.matmul(h_t_1, self.wa, name='matmul_2') + \
                     tf.matmul(slice_source, self.ua, name='matmul_3') + tf.matmul(slice_F, self.uc, name='matmul_4')
                a_t = tf.matmul(tf.tanh(ha), tf.expand_dims(self.va, -1), name='matmul_5')
                return tf.reshape(a_t, [-1])

            e_is = tf.scan(fn=att_Clcu, elems=[source_dist_major, F_flatten_dist_major], initializer=tf.zeros((batch_size,), tf.float32), name='scan_1')

            e_is = tf.stack(e_is, name='stack_1')
            e_is_transpose = tf.transpose(e_is)

            # (9) - (10)
            # masked attention
            attn_dist_t_flatten = masked_attention(e_is_transpose)
            attn_dist_t = tf.reshape(attn_dist_t_flatten, (-1, h, w, 1), name='reshape_4')
            c_t = tf.reduce_sum(source_reshape * tf.expand_dims(attn_dist_t_flatten, -1), axis=1)

            if is_training:
                Ey_t_1 = tf.nn.embedding_lookup(self.embedding, t_input)
            else:
                Ey_t_1 = tf.nn.embedding_lookup(self.embedding, inds_t_1)

            # GRU WAP (4)-(7)
            z_t = tf.nn.sigmoid(tf.matmul(Ey_t_1, self.yz, name='matmul_6')
                                + tf.matmul(h_t_1, self.uz, name='matmul_7') + tf.matmul(c_t, self.wz, name='matmul_8') + self.bz)
            r_t = tf.nn.sigmoid(tf.matmul(Ey_t_1, self.yr, name='matmul_9')
                                + tf.matmul(h_t_1, self.ur, name='matmul_10') + tf.matmul(c_t, self.wr, name='matmul_11') + self.br)
            h_proposal = tf.tanh(tf.matmul(Ey_t_1, self.yh, name='matmul_12')
                                 + tf.matmul(tf.multiply(r_t, h_t_1), self.uh, name='matmul_13') + tf.matmul(c_t, self.wh, name='matmul_14') + self.bh)
            h_t = tf.multiply(1 - z_t, h_t_1) + tf.multiply(z_t, h_proposal)

            # WAP (12)
            cover_t = cover_t_1 +  attn_dist_t

            # (11)
            logits_t = tf.matmul(Ey_t_1 + tf.matmul(h_t, self.ph, name='matmul_15') + tf.matmul(c_t, self.pc, name='matmul_16'), self.po, name='matmul_17') + self.pb
            softMax_t = tf.nn.softmax(logits_t, -1)
            max_inds_t = tf.argmax(softMax_t, -1, output_type=tf.int32)

            t_state = (h_t, cover_t, logits_t, max_inds_t, attn_dist_t)
            return t_state

        if is_training:
            elems = tf.transpose(iterator.target_input)
        else:
            assert tgt_sos_id is not None
            tgt_maxLen = int(cfg.tgt_max_len * cfg.allow_grow_ratio)
            tgt_maxLen = max(tgt_maxLen, cfg.tgt_max_len + 1)

            elems = tf.fill((tgt_maxLen, batch_size), tgt_sos_id)

        self.t_state = tf.scan(fn=forward_pass, elems=elems, initializer=t_0_state, name='scan_2')
        logits = tf.transpose(self.t_state[2], (1, 0, 2))
        indes = tf.transpose(self.t_state[3], (1, 0))
        attn_dists = tf.reshape(tf.transpose(self.t_state[4], (1, 0, 2, 3, 4)), (batch_size, -1, h, w))
        return logits, indes, attn_dists





