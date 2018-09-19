import numpy as np
import tensorflow as tf

def _shape(inputs, name=None):
    name = 'shape' if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)


def log(x, epsilon=1e-20, name=None):
    if isinstance(epsilon, float) and epsilon > 0:
        return tf.log(tf.maximum(x, epsilon), name=name)
    else:
        return tf.log(x, name=name)


def divide(x, y, epsilon=None, name=None):
    epsilon = 1e-20 if epsilon is None else epsilon
    name = 'safe_divide' if name is None else name
    with tf.name_scope(name):
        y = tf.where(tf.greater(tf.abs(y), epsilon), y, y + tf.sign(y) * epsilon)
        return tf.divide(x, y)
 

def transform(inputs, num_outputs, out_caps_dims, name=None):
    name = 'transform' if name is None else name
    with tf.variable_scope(name) as scope:
        input_shape = _shape(inputs)
        prefix_shape = [1 for i in range(len(input_shape) - 3)] + input_shape[-3:-2] + [num_outputs]
        in_caps_dims = input_shape[-2:]
        if in_caps_dims[0] == out_caps_dims[1]:
            shape = prefix_shape + [out_caps_dims[0], 1, in_caps_dims[1]]
            expand_axis = -3
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[0]:
            shape = prefix_shape + [in_caps_dims[0], 1, out_caps_dims[1]]
            expand_axis = -1
            reduce_sum_axis = -3
        elif in_caps_dims[0] == out_caps_dims[0]:
            shape = prefix_shape + [1, out_caps_dims[1], in_caps_dims[1]]
            expand_axis = -2
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[1]:
            shape = prefix_shape + [in_caps_dims[0], out_caps_dims[0], 1]
            expand_axis = -2
            reduce_sum_axis = -3
        else:
            raise TypeError('out_caps_dims must have at least one value being the same with the in_caps_dims')
        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=expand_axis)
        transform_mat = tf.get_variable('transform_mat', shape=shape)
        votes = tf.reduce_sum(in_pose * transform_mat, axis=reduce_sum_axis)
        return votes


def space_to_batch_nd(inputs, kernel_size, strides, name=None):
    name = 'space_to_batch_nd' if name is None else name
    with tf.name_scope(name):
        height, width, depth = _shape(inputs)[1:4]
        h_offset = [[(h+k) for k in range(0, kernel_size[0])] for h in range(0, height+1-kernel_size[0], strides[0])]
        w_offset = [[(w+k) for k in range(0, kernel_size[1])] for w in range(0, width+1-kernel_size[1], strides[1])]
        d_offset = [[(d+k) for k in range(0, kernel_size[2])] for d in range(0, depth+1-kernel_size[2], strides[2])]
        patched = tf.gather(inputs, h_offset, axis=1)
        patched = tf.gather(patched, w_offset, axis=3)
        patched = tf.gather(patched, d_offset, axis=5)

        if len(patched.shape) == 7:
            perm = [0, 1, 3, 5, 2, 4, 6]
        else:
            perm = [0, 1, 3, 5, 2, 4, 6, 7, 8]

        patched = tf.transpose(patched, perm=perm)
        shape = _shape(patched)

        if depth == kernel_size[2]: # for conv2d
            shape = shape[:3] + [np.prod(shape[3:-2])] + shape[-2:] if len(patched.shape) == 9 else shape[:3] + [np.prod(shape[3:])]
        else:
            shape = shape[:4] + [np.prod(shape[4:-2]) + shape[-2:]] if len(patched.shape) == 9 else shape[:4] + [np.prod(shape[4:])]

        patched = tf.reshape(patched, shape=shape)
        
    return patched


def softmax(logits, axis=None, name=None):
    name = 'Softmax' if name is None else name
    with tf.name_scope(name):
        if axis < 0:
            axis = len(logits.shape) + axis

        return tf.nn.softmax(logits, axis=axis)


def spread_loss(labels, logits, margin, regularizer=None):
    a_target = tf.reduce_sum(labels * logits, axis=1, keepdims=True)
    dist = (1 - labels) * margin - (a_target - logits)
    dist = tf.pow(tf.maximum(0., dist), 2)
    loss = tf.reduce_mean(tf.reduce_sum(dist, axis=-1))
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return(loss)