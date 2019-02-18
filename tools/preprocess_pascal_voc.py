"""
    PASCAL_VOC 数据预处理
"""
import os
import xml.etree.ElementTree as ET


classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

# 类别名称和ID映射
classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}

YOLO_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(YOLO_ROOT, 'data/VOCdevkit')
OUTPUT_PATH = os.path.join(YOLO_ROOT, 'data/pascal_voc.txt')

def parse_xml(xml_file):
  """
  解析 xml_文件

  输入：xml文件路径
  返回：图像路径和对应的label信息
  """
  # 使用ElementTree解析xml文件
  tree = ET.parse(xml_file)
  root = tree.getroot()
  image_path = ''
  labels = []

  for item in root:
    if item.tag == 'filename':
      image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', item.text)
    elif item.tag == 'object':
      obj_name = item[0].text
      # 将objetc的名称转换为ID
      obj_num = classes_num[obj_name]
      # 依次得到Bbox的左上和右下点的坐标
      xmin = int(item[4][0].text)
      ymin = int(item[4][1].text)
      xmax = int(item[4][2].text)
      ymax = int(item[4][3].text)
      labels.append([xmin, ymin, xmax, ymax, obj_num])

  # 返回图像的路径和label信息（Bbox坐标和类别ID）
  return image_path, labels

def convert_to_string(image_path, labels):
  """
     将图像的路径和lable信息转为string
  """
  out_string = ''
  out_string += image_path
  for label in labels:
    for i in label:
      out_string += ' ' + str(i)
  out_string += '\n'
  return out_string

def main():
  out_file = open(OUTPUT_PATH, 'w')

  # 获取所有的xml标注文件的路径
  xml_dir = DATA_PATH + '/VOC2007/Annotations/'
  xml_list = os.listdir(xml_dir)
  xml_list = [xml_dir + temp for temp in xml_list]

  # 解析xml文件，得到图片名称和lables，并转换得到图片的路径
  for xml in xml_list:
    try:
      image_path, labels = parse_xml(xml)
      # 将解析得到的结果转为string并写入文件
      record = convert_to_string(image_path, labels)
      out_file.write(record)
    except Exception:
      pass

  out_file.close()

if __name__ == '__main__':
  main()
