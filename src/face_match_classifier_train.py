
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import csv
from os.path import isdir, isfile


def is_image(image_name):
    try:
        ext = image_name.split('.')[-1]
        return ext.lower()=="jpg" or ext.lower()=="jpeg" or ext.lower()=="png"
    except:
        return False


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, '../data/')

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
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


def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # same person dist arr
            same_person_dists = []

            print('Loading images from directory: {}'.format(args.image_dir))
            for person_folder in os.listdir(args.image_dir):
                person_img_folder = os.path.join(args.image_dir, person_folder)

                # check is directory
                if not isdir(person_img_folder):
                    continue

                person_imgs =[x for x in os.listdir(person_img_folder) if is_image(x)]
                imgs_count = len(person_imgs)
                print('Found {} images from {}'.format(imgs_count, person_img_folder))
                for i in range(imgs_count-1):
                    img_1 = os.path.join(person_img_folder, person_imgs[i])
                    for j in range(i+1, imgs_count):
                        img_2 = os.path.join(person_img_folder, person_imgs[j])

                        # load, detect and align faces
                        images = load_and_align_data([img_1, img_2], args.image_size, args.margin, args.gpu_memory_fraction)

                        # Run forward pass to calculate embeddings
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
                        same_person_dists.append((img_1, img_2, dist))

            with open(args.output_file, 'a') as f:
                a = csv.writer(f, delimiter=',')
                a.writerows(same_person_dists)

            # nrof_images = len(args.image_files)
            #
            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, args.image_files[i]))
            # print('')
            #
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str, help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('image_dir', type=str, help='Images to compare')
    parser.add_argument('output_file', type=str, help='Output file for dist computation')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))