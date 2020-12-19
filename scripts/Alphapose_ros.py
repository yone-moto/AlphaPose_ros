#!/usr/bin/env python3
# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------


"""Script for single-image demo."""
import argparse
import torch
import os
import platform
import rospy
from sensor_msgs.msg import Image
import sys
try:
    print("remove path")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import argparse
import math
import time

import numpy as np

######################
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from threading import Thread
from queue import Queue
import torch.multiprocessing as mp
from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
######################

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime

from alphapose_ros.msg import AlphaPoseHumanList
from alphapose_ros.msg import AlphaPoseHuman

from sensor_msgs.msg import CompressedImage

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)


parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')

args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

class DetectionLoader():
    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, image):
        # start to pre process images for object detection
        self.image_preprocess(image)
        
        # start to detect human in images
        self.image_detection()
        # start to post process cropped human image for pose estimation
        self.image_postprocess()
        return self

    def image_preprocess(self, image):
        # expected image shape like (1,3,h,w) or (3,h,w)
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]


        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return
        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)
        boxes = boxes[dets[:, 0] == 0]

        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        self.det = (orig_imgs, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose



class DataWriter():
    def __init__(self, cfg, opt,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def wait_and_get(self, queue):
        return queue.get()

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        # keep looping infinitelyd
        # ensure the queue is not empty and get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, _) = self.wait_and_get(self.result_queue)
        # (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) =  self.item

        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            #pred = hm_data.cpu().data.numpy()

            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            pose_coords = []
            pose_scores = []
            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            if not self.opt.pose_track:
                boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)
            # boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            #     pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

            _result = []
            for k in range(len(scores)):

                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'result': _result
            }
            return result

    # def write_image(self, img, im_name, stream=None):
    #     if self.opt.vis:
    #         cv2.imshow("AlphaPose Demo", img)
    #         cv2.waitKey(30)
    #     if self.opt.save_img:
    #         cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
    #     if self.save_video:
    #         stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)


    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

class SingleImageAlphaPose():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

        self.pose_model.to(args.device)
        self.pose_model.eval()
        
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)
        if args.pose_track:
            self.tracker = Tracker(tcfg, self.args)

    def process(self, image):
        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, boxes, scores, ids, cropped_boxes) = self.det_loader.process(image).read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    self.writer.save(None, None, None, None, None, orig_img,None)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    pose = self.writer.update()
                    
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                else:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.args.device)
                    if self.args.flip:
                        inps = torch.cat((inps, flip(inps)))
                    hm = self.pose_model(inps)
                    # print(hm)
                    if self.args.flip:
                        hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                        hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    if args.pose_track:
                        im_name = " " 
                        boxes,scores,ids,hm,cropped_boxes = track(self.tracker,self.args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img,None)
                    pose = self.writer.update()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

            if self.args.profile:
                print(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            pass

        return pose

###############################


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]
    return color

class AlphaposeROS():
    def __init__(self):
        self.alphapose_list = AlphaPoseHumanList()
        self.alphapose_list.header.stamp = img_time
        self.alphapose_list.header.frame_id = FrameId

    def pub_compressed_image(self):
        img = CompressedImage()
        img.header.stamp = img_time
        img.format = "jpeg"
        img.data = np.array(cv2.imencode('.jpg', self.image_np)[1]).tostring() 
        image_pub.publish(img)

    def bbox_writer(self):
        cv2.rectangle(self.image_np, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3])), self.color)

    def id_writer(self):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(self.image_np,str(self.id_),(int(self.point[0]), int(self.point[1])), font, 2, (self.cl, 0, 0),2,cv2.LINE_AA)
        cv2.putText(self.image_np,str(self.id_),(int(self.bbox[0]), int((self.bbox[1]))), DEFAULT_FONT, 1, BLACK, 2)

    def point_writer(self):
        cv2.circle(self.image_np, (self.point[0], self.point[1]), 3,  self.color, thickness=-1)

    def cb_pose(self,pose,image_np):
        self.image_np = image_np
        for k in range(len(pose["result"])):
            alphapose = AlphaPoseHuman()
            self.bbox = pose["result"][k]["bbox"]
            self.id_ = pose["result"][k]["idx"]
            alphapose.id = self.id_
            self.color = get_color_fast(self.id_)
            
            pose_ = pose["result"][k]["keypoints"]
            alphapose.body_bounding_box.x = self.bbox[0]
            alphapose.body_bounding_box.y = self.bbox[1]
            alphapose.body_bounding_box.width = self.bbox[2]
            alphapose.body_bounding_box.height = self.bbox[3]

            for pose_num in range(len(pose_)):
                self.point = pose_[pose_num]
                alphapose.body_key_points_with_prob[pose_num].x = float(self.point[0])
                alphapose.body_key_points_with_prob[pose_num].y = float(self.point[1])
                self.point_writer()
            
            self.alphapose_list.human_list.append(alphapose)
            self.bbox_writer()
            self.id_writer()

        pub_pose.publish(self.alphapose_list)
        self.pub_compressed_image()
    
def cb_img(ros_data):
    global image_,img_time,FrameId
    img_time = ros_data.header.stamp
    FrameId = ros_data.header.frame_id
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_ = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    demo = SingleImageAlphaPose(args, cfg)
    rospy.init_node('alphapose_ros_node')
    rt = rospy.Rate(20)
    subscriber = rospy.Subscriber("/video_to_topic/image/compressed",CompressedImage, cb_img,  queue_size = 1)
    pub_pose = rospy.Publisher('/alphapose', AlphaPoseHumanList, queue_size=10)
    image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=10)

    while not rospy.is_shutdown():
        try:
            tmp_img = image_
            pose = demo.process(tmp_img)
            ros_pose = AlphaposeROS()
            ros_pose.cb_pose(pose,tmp_img)
        except:
            import traceback
            traceback.print_exc()
            print("no pose estimation")
            # try:
            #     make_img(image_)
            # except:
            #     pass
            
        rt.sleep()


###################################################