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

ImageDatabase = MongoClient()['Inception']['ImageData']


def cosine_similarity(vec1, vec2):
    num = np.dot(vec1, vec2)
    denorm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denorm == 0:
        return 0
    else:
        return num/denorm


def find_similar_pic(my_feature):
    image_data = ImageDatabase.find()
    max_sim = -1
    sim_dir = ''
    sim_file = ''
    for doc in image_data:
        feature = doc['feature']
        feature = np.array(feature)
        sim = cosine_similarity(my_feature, feature)
        if sim > max_sim:
            max_sim = sim
            sim_dir = doc['folder']
            sim_file = doc['number']
    return sim_dir, sim_file, max_sim


if __name__ == '__main__':
    with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for root, dir, files in os.walk('.\images'):
            for file in files:
                image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
                predictions = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})
                predictions.resize((2048,))
                sim_dir, sim_file, max_sim = find_similar_pic(predictions)
                print(file, sim_dir, sim_file, max_sim)
