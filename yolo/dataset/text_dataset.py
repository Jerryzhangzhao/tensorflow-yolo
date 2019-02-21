from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
import numpy as np
from queue import Queue 
from threading import Thread

from yolo.dataset.dataset import DataSet 

class TextDataSet(DataSet):
  """TextDataSet
     对数据预处理中得到的data list文件进行处理
     text file format:
     image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
  """

  def __init__(self, common_params, dataset_params):
    """
    Args:
      common_params: A dict
      dataset_params: A dict
    """
    #process params
    self.data_path = str(dataset_params['path'])
    self.width = int(common_params['image_size'])
    self.height = int(common_params['image_size'])
    self.batch_size = int(common_params['batch_size'])
    self.thread_num = int(dataset_params['thread_num'])
    self.max_objects = int(common_params['max_objects_per_image'])

    #定义两个队列，一个存放训练样本的list，另个存放训练样本的数据（image & label）
    self.record_queue = Queue(maxsize=10000)
    self.image_label_queue = Queue(maxsize=512)

    self.record_list = []  

    # 读取经过数据预处理得到的 pascal_voc.txt
    input_file = open(self.data_path, 'r')

    for line in input_file:
      line = line.strip()
      ss = line.split(' ')
      ss[1:] = [float(num) for num in ss[1:]]  # 将坐标和类别ID转为float
      self.record_list.append(ss)

    self.record_point = 0
    self.record_number = len(self.record_list)

    # 计算每个epoch的batch数目
    self.num_batch_per_epoch = int(self.record_number / self.batch_size)

    # 启动record_processor进程
    t_record_producer = Thread(target=self.record_producer)
    t_record_producer.daemon = True 
    t_record_producer.start()

    # 启动record_customer进程
    for i in range(self.thread_num):
      t = Thread(target=self.record_customer)
      t.daemon = True
      t.start() 

  def record_producer(self):
    """record_queue 的processor
    """
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list)
        self.record_point = 0
      # 从record_list读取一条训练样本信息到record_queue
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def record_process(self, record):
    """record 处理过程
    Args: record 
    Returns:
      image: 3-D ndarray
      labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
      object_num:  total object number  int 
    """
    image = cv2.imread(record[0])  # record[0]是image 的路径

    # 对图像做色彩空间变换和尺寸缩放
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h = image.shape[0]
    w = image.shape[1]

    width_rate = self.width * 1.0 / w 
    height_rate = self.height * 1.0 / h

    # 尺寸调整到 (448,448)
    image = cv2.resize(image, (self.height, self.width))

    labels = [[0, 0, 0, 0, 0]] * self.max_objects

    i = 1
    object_num = 0

    while i < len(record):
      xmin = record[i]
      ymin = record[i + 1]
      xmax = record[i + 2]
      ymax = record[i + 3]
      class_num = record[i + 4]
     
      # 由于图片缩放过，对label坐标做同样处理
      xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
      ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

      box_w = (xmax - xmin) * width_rate
      box_h = (ymax - ymin) * height_rate

      labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
      object_num += 1
      i += 5
      if object_num >= self.max_objects:
        break
    return [image, labels, object_num]

  def record_customer(self):
    """record queue的使用者
       取record queue中数据，经过处理后，送到image_label_queue中
    """
    while True:
      item = self.record_queue.get()
      out = self.record_process(item)
      self.image_label_queue.put(out)

  def batch(self):
    """获取一个batch的数据
    Returns:
      images: 4-D ndarray [batch_size, height, width, 3]
      labels: 3-D ndarray [batch_size, max_objects, 5]
      objects_num: 1-D ndarray [batch_size]
    """
    images = []
    labels = []
    objects_num = []
    for i in range(self.batch_size):
      image, label, object_num = self.image_label_queue.get()
      images.append(image)
      labels.append(label)
      objects_num.append(object_num)
    images = np.asarray(images, dtype=np.float32)
    images = images/255 * 2 - 1  # 将像素值转换到[-1,1]
    labels = np.asarray(labels, dtype=np.float32)
    objects_num = np.asarray(objects_num, dtype=np.int32)
    return images, labels, objects_num
