# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

# from nets.network import Network
from model.config import cfg

def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc


class selector(object):
  def __init__(self, mode):
    # Network.__init__(self)
    self.training = mode == 'TRAIN'
    self.testing = mode == 'TEST'
    self._scope = 'selector'
    self._decide_blocks()
    # H = W = 300
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def _build_base(self):
    with tf.variable_scope(self._scope, self._scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
    return net

  def select(self, num_select, reuse=None):
    assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
    # Now the base is always fixed during training
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      net_conv = self._build_base()
    if cfg.RESNET.FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                           self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=reuse,
                                           scope=self._scope)
    if cfg.RESNET.FIXED_BLOCKS < 3:
      with slim.arg_scope(resnet_arg_scope(is_training=self.training)):
        net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                           self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=reuse,
                                           scope=self._scope)
        net_conv = slim.conv2d(net_conv, num_outputs=128, kernel_size=[1, 1], padding="SAME", scope="c1_s1")
        net_conv = slim.conv2d(net_conv, num_outputs=num_select, kernel_size=[1, 1], padding="SAME", activation_fn=None, scope="c1_s2")
        # Selection; Find the selector index
        idx = tf.argmax(tf.nn.softmax(net_conv, -1), -1)
    return idx

  def _decide_blocks(self):
    # choose different blocks for different number of layers
    self._blocks = [resnet_v1_block('block1', base_depth=16, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=32, num_units=4, stride=2),
                  # use stride 1 for the last conv4 layer
                  resnet_v1_block('block3', base_depth=64, num_units=6, stride=1),
                  resnet_v1_block('block4', base_depth=128, num_units=3, stride=1)]
