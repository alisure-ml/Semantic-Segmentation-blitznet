# encoding:UTF-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
from os import path as osp
from glob import glob
import cv2
import scipy.misc as misc

from paths import DEMO_DIR
from detector import Detector
from config import args
from config import config as net_config
from resnet import ResNet

slim = tf.contrib.slim

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


class Loader(object):

    def __init__(self, folder=DEMO_DIR, data_format='.jpg'):
        cats = VOC_CATS
        self.folder = folder
        self.data_format = data_format
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]
        pass

    @staticmethod
    def deal_image(frame, what=True):
        if what:
            im = (np.array(frame) / 255.0).astype(np.float32)
        else:
            draw = Image.fromarray(np.asarray(frame, np.uint8), 'P')
            draw.putpalette(np.load('Extra/palette.npy').tolist())
            im = misc.fromimage(draw)
        return im

    pass


def main(argv=None):
    assert args.detect or args.segment, "Either detect or segment should be True"
    assert args.ckpt >= 0, "Specify the number of checkpoint"

    net = ResNet(config=net_config, depth=50, training=False)
    loader = Loader()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        detector = Detector(sess, net, loader, net_config, no_gt=args.no_seg_gt, folder=osp.join(loader.folder, 'output'))
        detector.restore_from_ckpt(args.ckpt)

        # 打开摄像头
        # cap = cv2.VideoCapture('video_demo/plane_1.mp4')
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)

        count = 0
        while cap.isOpened():
            begin_time = time.clock()
            ret, frame = cap.read()
            if ret:
                count += 1
                frame = cv2.flip(frame, 1)
                cv2.imshow("capture", frame)

                feed_forward_time = time.clock()
                image = loader.deal_image(frame, what=True)
                h, w = image.shape[:2]
                result = detector.feed_forward(img=image, name="video/{}".format(count), w=w, h=h, draw=False,
                                               seg_gt=None, gt_bboxes=None, gt_cats=None)[0]
                result = loader.deal_image(result, what=False)
                print("{} feed forward time is {}".format(count, time.clock() - feed_forward_time))
                cv2.imshow("Result", result)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
            print("{} frame time is {}".format(count, time.clock() - begin_time))
            pass

        cap.release()
        cv2.destroyAllWindows()

    pass


if __name__ == '__main__':
    tf.app.run()
