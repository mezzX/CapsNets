import numpy as np
import tensorflow as tf
import utils

def routing(votes, activation=None, run_iter=3, name=None):
    vote_rank = len(votes.shape)
    activation_rank = len(activation.shape)
    if vote_rank != 5 and vote_rank != 7:
        raise ValueError('Votes to routing should be rank 5 or 7, but it is rank ', str(vote_rank))
    if activation_rank != 2 and activation_rank != 4:
        raise ValueError('Activation to routing should be rank 2 or 4 but it is rank ', str(activation_rank))

    name = 'routing' if name is None else name
    with tf.variable_scope(name):
        pose, activation = EMRouting(votes, activation, run_iter)

    activation = tf.clip_by_value(activation, 1e-30, 1. - 1e-30)
    assert len(pose.shape) == 4 or len(pose.shape) == 6
    assert len(activation.shape) == 2 or len(activation.shape) == 4
    return(pose, activation)


def EMRouting(votes, activation, num_iter, bias=True):
    vote_shape = utils._shape(votes)
    num_outputs = vote_shape[-3]
    out_caps_dims = vote_shape[-2:]

    shape = vote_shape[:-2] + [np.prod(out_caps_dims)]
    votes = tf.reshape(votes, shape=shape)
    activation = activation[..., tf.newaxis, tf.newaxis]
    log_activation = tf.log(activation)
    log_R = tf.log(tf.fill(vote_shape[:-2] + [1], 1.) / num_outputs)

    lambda_min = 0.001
    lambda_max = 0.006
    for t_i in range(num_iter):
        inverse_tempt = lambda_min + (lambda_max - lambda_min) * t_i / max(1.0, num_iter)
        with tf.variable_scope('M_step') as scope:
            if t_i > 0:
                scope.reuse_variables()
            pose, log_var, log_act_prime = MStep(log_R, log_activation, votes, lambda_val=inverse_tempt)
            if t_i == num_iter - 1:
                break
        with tf.variable_scope('E_step'):
            log_R = EStep(pose, log_var, log_act_prime, votes)

    pose = tf.reshape(pose, shape=vote_shape[:-4] + [num_outputs] + out_caps_dims)
    activation = tf.reshape(tf.exp(log_act_prime), shape=vote_shape[:-4] + [num_outputs])
    return pose, activation


def MStep(log_R, log_activation, vote, lambda_val=0.01):
    R_shape = tf.shape(log_R)
    log_R = log_R + log_activation

    R_sum_i = tf.reduce_sum(tf.exp(log_R), axis=-3, keepdims=True)
    log_normal_R = log_R - tf.reduce_logsumexp(log_R, axis=-3, keepdims=True)

    pose = tf.reduce_sum(vote * tf.exp(log_normal_R), axis=-3, keepdims=True)
    log_var = tf.reduce_logsumexp(log_normal_R + utils.log(tf.square(vote - pose)), axis=-3, keepdims=True)

    beta_v = tf.get_variable('beta_v',
                              shape=[1 for i in range(len(pose.shape) - 2)] + [pose.shape[-2], 1],
                              initializer=tf.truncated_normal_initializer(mean=15., stddev=3.))
    cost = R_sum_i * (beta_v + 0.5 * log_var)

    beta_a = tf.get_variable('beta_a',
                              shape=[1 for i in range(len(pose.shape) - 2)] + [pose.shape[-2], 1],
                              initializer=tf.truncated_normal_initializer(mean=100.0, stddev=10))
    cost_sum_h = tf.reduce_sum(cost, axis=-1, keepdims=True)
    logit = lambda_val * (beta_a - cost_sum_h)
    log_activation = tf.log_sigmoid(logit)

    return(pose, log_var, log_activation)

def EStep(pose, log_var, log_activation, vote):
    normal_vote = utils.divide(tf.square(vote - pose), 2 * tf.exp(log_var))
    log_probs = normal_vote + utils.log(2 * np.pi) + log_var
    log_probs = -0.5 * tf.reduce_sum(log_probs, axis=-1, keepdims=True)
    log_act_logit = log_activation + log_probs
    log_act_logit = log_probs
    log_R = log_act_logit - tf.reduce_logsumexp(log_act_logit, axis=-2, keepdims=True)
    return log_R