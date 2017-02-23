from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import time
import argparse
import src.facenet as facenet
from src.align import detect_face
import csv
from os.path import isdir, isfile
import random
import logging
from settings import MODEL_DIR, IMG_PATH
from util.logger import create_logger


logger = create_logger('model', logging.DEBUG, 'model.log')


logger.info('Model directory: %s' % MODEL_DIR)
meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(MODEL_DIR))
logger.info('Metagraph file: %s' % meta_file)
logger.info('Checkpoint file: %s' % ckpt_file)

time_check_1 = time.time()
# set a facenet session
facenet_session = tf.Session()
facenet.load_model_with_session(facenet_session, MODEL_DIR, meta_file, ckpt_file)
time_check_2 = time.time()
logger.info("Loading facenet taken {} seconds".format(time_check_2-time_check_1))

# set a session for mtcnn face detection neural network
logger.info('Creating Multi-task Cascaded Convolutional Neural Networks and loading parameters')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
mtcnn_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with mtcnn_sess.as_default():
    pnet, rnet, onet = detect_face.create_mtcnn(mtcnn_sess, './data/')

time_check_3 = time.time()
logger.info("Loading MTCNN taken {} seconds".format(time_check_3-time_check_2))
logger.info("Get functions for face detection: {}, {}, {}".format(pnet, rnet, onet))


def compare_two_faces(face_1, face_2):

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # load, detect and align faces
    images = load_and_align_data([face_1, face_2], 160, 44, 1.0)

    # Run forward pass to calculate embeddings
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb = facenet_session.run(embeddings, feed_dict=feed_dict)

    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
    write_to_db(face_1, face_2, dist)

    # convert numpy.float32 to native python float
    return dist.item()


def write_to_db(face_1, face_2, dist):
    logger.info("Distance Compute | {}".format({face_1:face_1, face_2:face_2, dist:dist}))


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images