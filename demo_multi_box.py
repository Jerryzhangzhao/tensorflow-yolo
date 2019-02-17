import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


def process_predicts_multi_box(predicts):
    # predicts is a tensor with shape (N,grid_size,grid_size,30), 30=(4+1)*2+20
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]
    print(predicts.shape)

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))

    C_visual = np.reshape(C,(7,7,2))
    p_classes_visual = np.reshape(p_classes,(7,7,20))
    coordinate_visual = np.reshape(coordinate,(7,7,2,4))
   
    return C_visual,p_classes_visual,coordinate_visual

def coordinate_tranform(coordinate,grid_offset_x,grid_offset_y,grid_size,width,height):
    xcenter = coordinate[0] #this is a offset value normalized by grid width
    ycenter = coordinate[1]
    w = coordinate[2] #the w is the bbox width normalized by image width
    h = coordinate[3]

    xcenter = (grid_offset_x + xcenter) * grid_size
    ycenter = (grid_offset_y + ycenter) * grid_size

    w = w * width
    h = h * height

    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0
    
    xmax = xmin + w
    ymax = ymin + h
    return xmin, ymin, xmax, ymax
    

common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

# network,input place holder and output tensor
net = YoloTinyNet(common_params, net_params, test=True)
image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

np_img = cv2.imread('cat2.jpeg')

# image preprocess
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
np_img = np_img.astype(np.float32)
np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

# load model and reference
saver = tf.train.Saver()
saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
np_predict = sess.run(predicts, feed_dict={image: np_img})

#xmin, ymin, xmax, ymax, class_num = process_predicts_multi_box(np_predict)
C_visual,p_classes_visual,coordinate_visual = process_predicts_multi_box(np_predict)

for i in range(7):
    for j in range(7):
        object_index = np.argmax(C_visual[i][j])
        if C_visual[i][j][object_index] > 0.4:
            coordinate = coordinate_visual[i][j][object_index]
            class_index = np.argmax(p_classes_visual[i][j])
            class_name = classes_name[class_index]
            if p_classes_visual[i][j][class_index] > 0.5:
                xmin, ymin, xmax, ymax = coordinate_tranform(coordinate,i,j,448/7.0,448,448)
                cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                print(coordinate)

cv2.imwrite('cat_out.jpg', resized_img)
sess.close()
