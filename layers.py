import numpy as np
import tensorflow as tf
import routing
import utils

def primaryCaps(inputs, filters, kernel_size, strides, out_caps_dims, name=None):
    name = 'primary_capsule' if name is None else name
    with tf.variable_scope(name):
        channels = filters * np.prod(out_caps_dims) + filters
        pose = tf.layers.conv2d(inputs,
                                channels,
                                kernel_size=kernel_size,
                                strides=strides, 
                                activation=None)
        shape = utils._shape(pose, name='get_pose_shape')
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        shape = [batch_size, height, width, filters] + out_caps_dims

        pose, activation_logit = tf.split(pose, [channels - filters, filters], axis=-1)
        pose = tf.reshape(pose, shape=shape)
        activation = tf.sigmoid(activation_logit)
        activation = tf.clip_by_value(activation, 1e-20, 1. - 1e-20)

        return(pose, activation)


def conv2d(inputs, activation, filters, out_caps_dims, kernel_size,
           strides, padding='valid', name=None, reuse=None):
    name = 'conv2d' if name is None else name
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input_shape = utils._shape(inputs)
        input_rank = len(input_shape)
        activation_rank = len(activation.shape)

        if not input_rank == 6:
            raise ValueError('Inputs to conv2d should have rank 6. Got inputs rank: ', str(input_rank))
        if not activation_rank == 4:
            raise ValueError('Activation to conv2d should have rank 4 Got activation rank: ', str(activation_rank))

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size, input_shape[3]]
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
            kernel_size = [kernel_size[0], kernel_size[1], input_shape[3]]
        else:
            raise ValueError('kernel_size should be an interger or tuple/list of 2 intergers. Got: ', str(kernel_size))

        if isinstance(strides, int):
            strides = [strides, strides, 1]
        elif isinstance(strides, (list, tuple)) and len(strides) == 2:
            strides = [strides[0], strides[1], 1]
        else:
            raise ValueError('strides should be an interger or tuple/list of 2 intergers. Got: ', str(strides))

        if not isinstance(out_caps_dims, (list, tuple)) or len(out_caps_dims) != 2:
            raise ValueError('out_caps_dims should be a tuple/list of 2 intergers. Got: ', str(out_caps_dims))
        elif isinstance(out_caps_dims, tuple):
            out_caps_dims = list(out_caps_dims)

        batched = utils.space_to_batch_nd(inputs, kernel_size, strides)
        activation = utils.space_to_batch_nd(activation, kernel_size, strides)

        vote = utils.transform(batched, num_outputs=filters, out_caps_dims=out_caps_dims)

        pose, activation = routing.routing(vote, activation)

        return pose, activation


def conv1d():
    #TODO
    pass


def dense(inputs, activation, num_outputs, out_caps_dims, coord_addition=False, reuse=None, name=None):
    name = 'dense' if name is None else name
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse()
        if coord_addition and len(inputs.shape) == 6 and len(activation.shape) == 4:
            vote = utils.transform(inputs, num_outputs=num_outputs, out_caps_dims=out_caps_dims)
            with tf.name_scope('coord_addition'):
                batch_size, input_h, input_w, input_c, _, out_caps_h, out_caps_w = utils._shape(vote)
                num_inputs = input_h * input_w * input_c

                zeros = np.zeros((input_h, out_caps_w - 1))
                coord_offset_h = ((np.arange(input_h) + 0.5) / input_h).reshape([input_h, 1])
                coord_offset_h = np.concatenate([zeros, coord_offset_h], axis=-1)
                zeros = np.zeros((out_caps_h - 1, out_caps_w))
                coord_offset_h = np.stack([np.concatenate([coord_offset_h[i:(i + 1), :], zeros], axis=0) for i in range(input_h)], axis=0)
                coord_offset_h = coord_offset_h.reshape((1, input_h, 1, 1, 1, out_caps_h, out_caps_w))

                zeros = np.zeros((1, input_w))
                coord_offset_w = ((np.arange(input_w) + 0.5) / input_w).reshape([1, input_w])
                coord_offset_w = np.concatenate([zeros, coord_offset_w, zeros, zeros], axis=0)
                zeros = np.zeros((out_caps_h, out_caps_w -1))
                coord_offset_w = np.stack([np.concatenate([zeros, coord_offset_w[:, i:(i+1)]], axis=1) for i in range(input_w)], axis=0)
                coord_offset_w = coord_offset_w.reshape((1, 1, input_w, 1, 1, out_caps_h, out_caps_w))

                vote = vote + tf.constant(coord_offset_h + coord_offset_w, dtype=tf.float32)
                vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs] + out_caps_dims)
                activation = tf.reshape(activation, shape=[batch_size, num_inputs])

        elif len(inputs.shape) == 4 and len(activation.shape) == 2:
            vote = utils.transform(inputs, num_outputs=num_outputs, out_caps_dims=out_caps_dims)

        else:
            raise TypeError('Wrong rank for inputs or activation')

        pose, activation = routing.routing(vote, activation)
        assert len(pose.shape) == 4
        assert len(activation.shape) == 2

    return(pose, activation)