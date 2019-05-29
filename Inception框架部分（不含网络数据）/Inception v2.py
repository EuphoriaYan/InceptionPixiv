# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from pymongo import MongoClient


if __name__ == '__main__':
    with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        image_predicitons = open('imagedata.txt', 'w')
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        # 遍历目录
        for root1, dirs1, files1 in os.walk('G:\PixivSpider\original_images'):
            for dir1 in dirs1:
                for root2, dirs2, files2 in os.walk('G:\PixivSpider\original_images\\' + dir1):
                    for file in files2:
                        image_data = tf.gfile.FastGFile(os.path.join(root2, file), 'rb').read()
                        predictions = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})
                        predictions = np.squeeze(predictions).tolist()
                        image_predicitons.write(dir1 + '\n')
                        image_predicitons.write(file + '\n')
                        image_predicitons.write(str(predictions) + '\n')
        image_predicitons.close()
'''
    image_database = MongoClient()['Inception']['ImageData']
    image_data = open('.\imagedata.txt', 'w')
    lines = image_data.readlines()
    doc = {
        '_id': '',
        'folder': '',
        'number': '',
        'feature': ''
    }
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        if i % 3 == 0:
            doc['folder'] = line
'''