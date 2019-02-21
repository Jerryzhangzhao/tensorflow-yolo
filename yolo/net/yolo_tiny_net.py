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
    # 将Bbox坐标由(x_center,y_center,w,h) 转为 (x_min, y_min, x_max, y_max)
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    # 上面这两句stack+transpose的操作也可以写成一句:
    # boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
    #                   boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2],axis=3)

    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    # 计算重合区域的左上和右下顶点
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    # 计算重叠区域面积
    intersection = rd - lu
    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
    # predict box和label box也可能没有重叠区域，这里的mask=0时候就是没有重叠区域的情况
    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    # 分别计算predict box和label box各自的面积
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    # 计算并返回IoU的值，返回的tensor的shape 是(cell_size,cell_size,box_pre_cell) 如7*7*2
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def cond1(self, num, object_num, loss, predict, label):
    """
       num初始值为0
       依次处理每个object
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels):
    """
    每次计算一张图片中的一个object的损失
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
    """
    label = labels[num:num+1, :]   # 取第num个object的label：(x_center, y_center, w, h, class)
    label = tf.reshape(label, [-1])

    # ==1==.计算有物体的那些格子坐标，即标记出物体覆盖到的那些格子（用于计算物体检测损失）
    # 根据label的坐标[x_center, y_center, w, h]和格子的数目计算以格子坐标表示的坐标值
    min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
    max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)
    min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
    max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

    # 分别取整得到格子坐标
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
    # paddings的shape为[n,2]，n为待填充的tensor的秩，‘CONSTANT’表示使用0填充
    objects = tf.pad(objects, paddings, "CONSTANT")

    # ==2==.使用label Bbox计算responsible tensor，实际上是标记出物体中心所在的格子 （用于计算坐标损失）
    # 将label Bbox的中心由像素坐标转为格子坐标
    center_x = label[0] / (self.image_size / self.cell_size)
    center_x = tf.floor(center_x)
    center_y = label[1] / (self.image_size / self.cell_size)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    response = tf.pad(response, temp, "CONSTANT")

    # ==3==.计算预测Bbox和label Bbox的IoU iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    # predict的shape为:[cell_size,cell_size,class_num+box_num*5]
    # 这里需要明确网络预测(inference方法)的返回predict中的坐标是‘偏移+归一化后’的还是像素坐标，即明确其格式，在预测推理的时候要根据其格式‘转换’坐标值；
    # 在下面第三行predict_boxes = predict_boxes * [self ... 这一行代码中可以看到对predict坐标做了一个‘反归一化和偏移的’计算；
    # 所以网络输出的坐标确实是‘偏移+归一化’后的格式
    # 因为在这里对坐标进行了‘反偏移和归一化’，所以在计算坐标损失的时候又重新进行了一次‘偏移和归一化’的步骤

    predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]

    predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

    # 将偏移+归一化的predict_boxes 由[x_offset_norm,y_offset_norm,w_norm,h_norm] 转换为[x,y,w,h](单位为像素值)
    # 1)‘反归一化’
    predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

    # 2)‘反偏移’
    # base_boxes 表示的是每个格子的坐标对应在图像中的像素坐标
    base_boxes = np.zeros([self.cell_size, self.cell_size, 4])
    for y in range(self.cell_size):
      for x in range(self.cell_size):
        base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
    
    # 扩展为2个Bbox
    base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])   
    
    # 将predict_boxes 由[x_offset,y_offset,w,h](单位为像素值)转换为[x,y,w,h](单位为像素值)
    predict_boxes = base_boxes + predict_boxes

    # 计算IoU,返回的iou_predict_truth的shape为(cell_size,cell_size,box_pre_cell)
    iou_predict_truth = self.iou(predict_boxes, label[0:4])

    # C tensor:responsible格子（物体中心落在的那个格子）的两个Bbox的IoU值，shape： [cell_size, cell_size, boxes_per_cell]
    C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

    # I tensor:responsible格子（物体中心落在的那个格子）的两个Bbox的IoU值，shape： [cell_size, cell_size, boxes_per_cell]
    I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))
    # 获取最大的IoU的值, max_I的shape: (cell_size,cell_size,1)
    max_I = tf.reduce_max(I, 2, keep_dims=True)

    # 这里的 I 的shape是(cell_size,cell_size,box_per_cell)，其含义是IoU最大的那个Bbox在tensor中的位置，所在位置为1,其他为0
    # 经过这一步，也就得到了文章中说的'the jth bounding box predictor in cell i is “responsible”for that prediction'
    # 也就是物体中心所落在的那个格子给出的N预测Bboxes中与label_box之间IoU最大的那个Bbox
    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    # no_I是与I的shape相同，但取值相反的tensor
    # 这一步得到了文章中的noobj
    no_I = tf.ones_like(I, dtype=tf.float32) - I 

    # p_C 这里是Bbox中有物体的概率
    p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

    # ==4== 计算Loss
    # （1）准备计算坐标损失的相关数据
    x = label[0]
    y = label[1]
    # 文章中在计算坐标损失的w，h项作了开平方缩放
    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))

    # predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

    # （2）准备计算类别损失的相关数据
    # 将lebel中的类别ID转为one_hot编码
    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:self.num_classes]

    # （3）分别计算类别损失、物体检测损失和坐标损失
    # 类别损失(class_loss)
    # 每个cell会给出N个预测的Bbox，比如2个，但是只有一组物体类别的概率
    # 计算类别损失的时候只计算出现了物体的那些格子的损失，所以这里用到了objects
    # class_scale 是类别损失的权重，论文中的Loss公式没有写出这个参数，默认为1,实际上在train.cfg中class_scale设置的是1.
    class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    # 物体检测loss(object_loss & noobject_loss)
    # 物体检测loss分成两类，一是responsible的那一个Bbox，称为object_loss，二是其他的Bbox,称为noobject_loss
    # 这里计算损失的时候用p_C - C，p_C是模型预测的Bbox中有无物体的概率，C是物体中心所在的那个格子的的Bbox的IoU值
    # 这里实际山是用IoU的值代替有无物体的ground_truth值
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale

    # noobject_loss
    # 对于这些‘noobject’的Bbox，理想的情况下是将他们都预测为无物体，也就是p_C值越小越好
    # 所以这里可以直接使用预测的Bbox有物体的概率p_C来计算损失
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    # 坐标损失(coord_loss)
    # 计算坐标损失的时候，对格子中心坐标用的时候中心相对于所在格子左上角的偏移量并以格子宽度进行归一化后的值
    # 对宽高用的是原始宽高使用图片宽高进行归一化后的值
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                 tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels



  def loss(self, predicts, labels, objects_num):
    """计算Loss
    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    # 损失函数由三部分构成：类别损失，物体检测损失（有物体，无物体），Bbox坐标损失
    class_loss = tf.constant(0, tf.float32)    # 类别损失
    object_loss = tf.constant(0, tf.float32)   # 有物体的损失
    noobject_loss = tf.constant(0, tf.float32) # 无物体的损失
    coord_loss = tf.constant(0, tf.float32)    # 坐标损失
    loss = [0, 0, 0, 0]
    for i in range(self.batch_size):
      predict = predicts[i, :, :, :] # 每张图片的prediction tensor
      label = labels[i, :, :]
      object_num = objects_num[i] # 图片中的物体的数目

      # 关于tf.while_loop(cond, body, var)
      # loop（var 中满足cond的条件，带入body计算），loop结束，返回结果。
      # >>> i = tf.constant(0)
      # >>> c = lambda i: tf.less(i, 10)
      # >>> b = lambda i: tf.add(i, 1)
      # >>> r = tf.while_loop(c, b, [i])

      # 这里的while_loop 循环的是多个object
      tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss], predict, label])

      for j in range(4):
        loss[j] = loss[j] + tuple_results[2][j]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size)

    # 添加到summary
    tf.summary.scalar('class_loss', loss[0]/self.batch_size)
    tf.summary.scalar('object_loss', loss[1]/self.batch_size)
    tf.summary.scalar('noobject_loss', loss[2]/self.batch_size)
    tf.summary.scalar('coord_loss', loss[3]/self.batch_size)
    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size )

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
