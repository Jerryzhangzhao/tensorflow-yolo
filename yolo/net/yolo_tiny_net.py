from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net 

class YoloTinyNet(Net):

  def __init__(self, common_params, net_params, test=False):
    """
    common params: a params dict
    net_params   : a params dict
    """
    super(YoloTinyNet, self).__init__(common_params, net_params)
    #process params
    self.image_size = int(common_params['image_size'])
    self.num_classes = int(common_params['num_classes'])
    self.cell_size = int(net_params['cell_size'])
    self.boxes_per_cell = int(net_params['boxes_per_cell'])
    self.batch_size = int(common_params['batch_size'])
    self.weight_decay = float(net_params['weight_decay'])

    if not test:
      # define the loss wright of different kind of loss
      self.object_scale = float(net_params['object_scale'])
      self.noobject_scale = float(net_params['noobject_scale'])
      self.class_scale = float(net_params['class_scale'])
      self.coord_scale = float(net_params['coord_scale'])

  def inference(self, images):
    """构建yolo_tiny网络

    输入：
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    返回:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    conv_num = 1

    temp_conv = self.conv2d('conv' + str(conv_num), images, [3, 3, 3, 16], stride=1)
    conv_num += 1

    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 16, 32], stride=1)
    conv_num += 1

    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 32, 64], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1

    temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2)) #(N,H,W,C)=>(N,C,H,W)

    # 全链接层
    local1 = self.local('local1', temp_conv, self.cell_size * self.cell_size * 1024, 256)
    local2 = self.local('local2', local1, 256, 4096)
    local3 = self.local('local3', local2, 4096, self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5), leaky=False, pretrain=False, train=True)

    # 对全连接层输出的tensor进行reshape
    # 全连接输出的长度cell_size*cell_size*(num_class+boxes_per_cell*5)二维tensor（还有一个维度是图片数目N）
    # YOLO论文中的7*7*(20+5*2)

    # 这里对local3进行reshape时，先将class_prob，objectness_prob和coordinate分别取出，各自reshape，最后合并到一起
    # 这样最后得到的tensor的各个通道是按照class_prob，objectness_prob和coordinate排列的
    n1 = self.cell_size * self.cell_size * self.num_classes
    n2 = n1 + self.cell_size * self.cell_size * self.boxes_per_cell

    class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes))  #class_prob
    scales = tf.reshape(local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell))   #objectness_prob
    boxes = tf.reshape(local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))  #coordinate

    # 合并得到输出 [N,cell_size,cell_size,class_num+bbox_num*5]
    local3 = tf.concat([class_probs, scales, boxes], axis=3)

    predicts = local3

    return predicts

  def iou(self, boxes1, boxes2):
    """IoU 计算
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    #calculate the left up point
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection
    intersection = rd - lu 

    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
    # inter_quare with shape [CELL_SIZE,CELL_SIZE,BOX_NUM]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def cond1(self, num, object_num, loss, predict, label, nilboy):
    """
       num初始值为0
       依次处理每个object
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels, nilboy):
    """
    计算损失
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
    """
    label = labels[num:num+1, :]   # 取第num个object的label：(x_center, y_center, w, h, class)
    label = tf.reshape(label, [-1])

    # 1.计算有物体的那些格子坐标，即标记出物体覆盖到的那些格子（用于计算物体检测损失）
    # 将label的坐标由[x_center, y_center, w, h]转为以格子坐标表示的坐标值
    min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
    max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)
    min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
    max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

    # 分别取整
    min_x = tf.floor(min_x)
    min_y = tf.floor(min_y)
    max_x = tf.ceil(max_x)
    max_y = tf.ceil(max_y)

    # objects与格子中有图像的区域大小一致，元素的值都为1
    temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    objects = tf.ones(temp, tf.float32) 

    # paddings是为了将objects扩展到与格子一样大小，所需在objects的四周需要padding的格子数目，顺序为top,down,left,right
    paddings = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)

    paddings = tf.reshape(paddings, (2, 2))
    # 这里得到的objects就是一个‘尺寸’为cell_size*cell_size,并且有物体的区域标为1,无物体区域标为0
    # 参数 CONSTANT 表示用0填充
    objects = tf.pad(objects, paddings, "CONSTANT")

    # 2.计算responsible tensor，实际上是标记出物体中心所在的格子 （用于计算坐标损失）
    # 将Bbox的中心坐标转为格子坐标
    center_x = label[0] / (self.image_size / self.cell_size)
    center_x = tf.floor(center_x)
    center_y = label[1] / (self.image_size / self.cell_size)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    response = tf.pad(response, temp, "CONSTANT")

    # 3.计算 iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    # shape of predict:[CELL_SIZE,CELL_SIZE,CLASS_NUM+BOX_NUM+BOX_NUM*4]
    predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:] # box坐标，坐标是相对当前格子左上角的偏移量归一化值

    predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

    # 将偏移+归一化的predict_boxes 由[x_offset_norm,y_offset_norm,w_norm,h_norm] 转换为[x_offset,y_offset,w,h](单位为像素值)
    predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

    # base_boxes 表示的是每个格子的坐标对应在图像中的像素坐标
    base_boxes = np.zeros([self.cell_size, self.cell_size, 4])
    for y in range(self.cell_size):
      for x in range(self.cell_size):
        #nilboy
        base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
    
    # expand to 2 boxes
    base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])   
    
    # 将predict_boxes 由[x_offset,y_offset,w,h](单位为像素值)转换为[x,y,w,h](单位为像素值)
    predict_boxes = base_boxes + predict_boxes

    # ==now we got predict boxes with coordinate x,y,w,h with pixel scale==

    iou_predict_truth = self.iou(predict_boxes, label[0:4]) 
    # iou_predict_truth with shape [CELL_SIZE,CELL_SIZE,BOX_NUM],and the element is IoU value


    #calculate C [cell_size, cell_size, boxes_per_cell]
    C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1)) 
    
    max_I = tf.reduce_max(I, 2, keep_dims=True) #get the box with max IoU

    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I 


    p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

    #calculate truth x,y,sqrt_w,sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))


    #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

    #calculate truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:self.num_classes]

    #class_loss
    class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale #p_C-C ,C is the IoU value,can be treat as the obejctness score
    #object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

    #noobject_loss
    #noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    #coord_loss
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                 tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    nilboy = I

    return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels, nilboy



  def loss(self, predicts, labels, objects_num):
    """计算Loss值

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    # 损失函数由三部分构成：类别损失，物体检测损失（有物体，无物体），Bbox坐标损失
    class_loss = tf.constant(0, tf.float32)    # 类别损失
    object_loss = tf.constant(0, tf.float32)   # 有物体的损失
    noobject_loss = tf.constant(0, tf.float32) # 非物体的损失
    coord_loss = tf.constant(0, tf.float32)    # 坐标损失
    loss = [0, 0, 0, 0]
    for i in range(self.batch_size):
      predict = predicts[i, :, :, :] # 每张图片的prediction tensor
      label = labels[i, :, :]
      object_num = objects_num[i] # 图片中的物体的数目
      nilboy = tf.ones([7,7,2])
      # tf.while_loop(cond, body, var)
      # loop（var 中满足cond的条件，带入body计算），loop结束，返回结果。
      # >>> i = tf.constant(0)
      # >>> c = lambda i: tf.less(i, 10)
      # >>> b = lambda i: tf.add(i, 1)
      # >>> r = tf.while_loop(c, b, [i])

      tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss], predict, label, nilboy])
      for j in range(4):
        loss[j] = loss[j] + tuple_results[2][j]
      nilboy = tuple_results[5]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size)

    tf.summary.scalar('class_loss', loss[0]/self.batch_size)
    tf.summary.scalar('object_loss', loss[1]/self.batch_size)
    tf.summary.scalar('noobject_loss', loss[2]/self.batch_size)
    tf.summary.scalar('coord_loss', loss[3]/self.batch_size)
    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size )

    return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy
