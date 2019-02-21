from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from yolo.solver.solver import Solver

class YoloSolver(Solver):
  """Yolo Solver 
  """
  def __init__(self, dataset, net, common_params, solver_params):
    #process params
    self.moment = float(solver_params['moment'])
    self.learning_rate = float(solver_params['learning_rate'])
    self.batch_size = int(common_params['batch_size'])
    self.height = int(common_params['image_size'])
    self.width = int(common_params['image_size'])
    self.max_objects = int(common_params['max_objects_per_image'])
    self.pretrain_path = str(solver_params['pretrain_model_path'])
    self.train_dir = str(solver_params['train_dir'])
    self.max_iterators = int(solver_params['max_iterators'])

    self.dataset = dataset
    self.net = net
    #construct graph
    self.construct_graph()

  def _train(self):
    """训练模型
    创建优化器，最小化Loss
    Args:
      total_loss: Total loss from net.loss()
      global_step: Integer Variable counting the number of training steps
      processed
    Returns:
      train_op: op for training
    """
    # 使用Momentum优化算法
    opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
    grads = opt.compute_gradients(self.total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

    # 这里也可以直接写成
    # tf.train.MomentumOptimizer(self.learning_rate,self.moment).minimize(self.total_loss,global_step=self.global_step)

    return apply_gradient_op

  def construct_graph(self):
    # 构建graph
    self.global_step = tf.Variable(0, trainable=False)
    # (1)训练时网络的输入
    self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
    self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
    self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

    # (2)inference部分，输入是一张图片，输出是一个(N,cell_size,cell_size,class_num+box_num*5)的tensor
    self.predicts = self.net.inference(self.images)

    # (3)loss 部分
    self.total_loss = self.net.loss(self.predicts, self.labels, self.objects_num)
    
    tf.summary.scalar('loss', self.total_loss)
    self.train_op = self._train()

  def solve(self):
    saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=1)
    saver2 = tf.train.Saver(self.net.trainable_collection, write_version=1)

    # 变量初始化
    init =  tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(init)

    # 加载预训练模型
    saver1.restore(sess, self.pretrain_path)

    # 创建 event file writer
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

    for step in range(self.max_iterators):
      start_time = time.time()
      # 获取一个batch的训练数据
      np_images, np_labels, np_objects_num = self.dataset.batch()

      _, loss_value = sess.run([self.train_op, self.total_loss], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})


      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = self.dataset.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

        sys.stdout.flush()
      if step % 100 == 0: # 保存event file
        summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
        summary_writer.add_summary(summary_str, step)
      if step % 5000 == 0: # 保存checkpoint
        saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)
    sess.close()

