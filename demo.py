import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

# 对网络给出的预测结果做处理
def process_predicts(predicts):
    # predicts 的shape是 (N,grid_size,grid_size,30), 30=(4+1)*2+20
    p_classes = predicts[0, :, :, 0:20] # 类别的概率
    C = predicts[0, :, :, 20:22]        # Bbox中有物体的概率
    coordinate = predicts[0, :, :, 22:] # 预测的Bbox坐标
    print(predicts.shape)

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))

    # P = 有物体的概率 * 类别的概率
    P = C * p_classes
    print(P.shape)

    # 找到有最大的概率P的Bbox
    index = np.argmax(P)
    index = np.unravel_index(index, P.shape)

    class_num = index[3]

    coordinate = np.reshape(coordinate, (7, 7, 2, 4))

    max_coordinate = coordinate[index[0], index[1], index[2], :]

    # 对网络输出的坐标值进行处理
    # 网络输出的Bbox的中心坐标是相对于格子左上角的坐标，并且用格子的宽度进行归一化(偏移+归一化)，这里需要处理成在原图中的坐标
    # 网络输出的Bbox的宽高是相对于图片大小归一化的，这里也要恢复成原始大小
    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]

    # ‘恢复’中心坐标：反偏移，反归一化
    xcenter = (index[1] + xcenter) * (448/7.0)
    ycenter = (index[0] + ycenter) * (448/7.0)
    # ‘恢复’宽高到原始像素大小
    w = w * 448
    h = h * 448

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0

    xmax = xmin + w
    ymax = ymin + h

    # 这里检测部分写的比较‘简单’，直接取了物体概率*类别概率最大的那个Bbox和class的结果
    # 实际上应该对每一个类分别进行检测，并用NMS去除多余的候选框
    return xmin, ymin, xmax, ymax, class_num


common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

# network,input place holder and output tensor
net = YoloTinyNet(common_params, net_params, test=True)
image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

# 读入图片
np_img = cv2.imread('cat.jpg')
height, width, channels = np_img.shape
print(height, width, channels)


# 对图片作处理，尺寸缩放,值映射到[-1,1]
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
np_img = np_img.astype(np.float32)
np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

# 加载模型，并做前向传播得到检测结果
saver = tf.train.Saver()
saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
np_predict = sess.run(predicts, feed_dict={image: np_img})

xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', resized_img)
sess.close()
