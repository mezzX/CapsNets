import tensorflow as tf

def image(name, tensor, verbose=True, max_outputs=3, collections=None, family=None):
    if verbose:
        tf.summary.image(name, tensor, max_outputs, collections, family)
    else:
        pass


def scalar(name, tensor, verbose=False, collections=None, family=None):
    if verbose:
        tf.summary.scalar(name, tensor, collections, family)
    else:
        pass


def histogram(name, values, verbose=True, collections=None, family=None):
    if verbose:
        tf.summary.histogram(name, values, collections, family)
    else:
        pass


def tf_stats(name, tensor, verbose=True, collections=None, family=None):
    if verbose:
        with tf.name_scope(name):
            mean = tf.reduce_mean(tensor)
            tf.summary.scalar('mean', mean, collections=collections, family=family)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))

        tf.summary.scalar('stddev', stddev, collections=collections, family=family)
        tf.summary.scalar('max', tf.reduce_max(tensor), collections=collections, family=family)
        tf.summary.scalar('min', tf.reduce_min(tensor), collections=collections, family=family)
        tf.summary.histogram('histogram', tensor, collections=collections, family=family)
    else:
        pass