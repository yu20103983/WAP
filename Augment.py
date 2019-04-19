#coding=utf-8
import tensorflow as tf

def nonlinear(imageList, lower, upper):
    with tf.name_scope('nonlinear') as scope:
        factor = tf.random_uniform([], lower, upper)

        res = []
        for i in imageList:
            res.append(tf.pow(i, factor))

        return res


def randomNormal(imageList, stddev):
    with tf.name_scope('randomNormal') as scope:
        factor = tf.random_uniform([], 0, stddev)

        res = []
        for i in imageList:
            res.append(i + tf.random_normal(tf.shape(i), mean=0.0, stddev=factor))

        return res


def mirror(image):
    uniform_random = tf.random_uniform([], 0, 1.0)
    return tf.cond(uniform_random < 0.5, lambda: image, lambda: tf.reverse(image, axis=[2]))


def augment(image):
    with tf.name_scope('augmentation') as scope:
        image = nonlinear([image], 0.8, 1.2)[0]  # 乘上一个随机因子

        # image = mirror(image)  # 镜像翻转

        image = tf.image.random_contrast(image, lower=0.3, upper=1.3)  # 随机调整对比度
        image = tf.image.random_brightness(image, max_delta=0.3)  # 随机调整亮度

        image = randomNormal([image], 0.025)[0]  # 随机噪音

        image = tf.clip_by_value(image, 0, 1.0)  # 剪切

        return image