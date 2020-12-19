#!/usr/bin/env python3
import pandas as pd
from utils.utils3 import make_std
import numpy as np
import glob
import pickle
import rospy
from alphapose_ros.msg import AlphaPoseHumanList
from alphapose_ros.msg import AlphaPoseHuman
from alphapose_ros.msg import PredHumanVelocityList
from alphapose_ros.msg import PredHumanVelocity

class PredictModel():
    def __init__(self,filename = "/home/robog/catkin_ws/src/alphapose_ros/scripts/models/RosModel.sav"):
        self.maxlen = 10
        self.input_list = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder"
                        ,"LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee",
                        "RKnee","LAnkle","RAnkle", "Head", "Neck", "Hip","LBigToe",
                        "RBigToe","LSmallToe","RSmallToe","LHeel","RHeel" ]
        self.filename = filename
        # モデルをロードする
        self.loaded_model = pickle.load(open(self.filename, 'rb'))
        self.cols = []
        for point in self.input_list:
            self.cols.append(f"x_{point}")
            self.cols.append(f"y_{point}")
            
        self.df = pd.DataFrame(index=[], columns=self.cols)
        init_index = [0]*len(self.cols)
        record = pd.Series(init_index, index=self.cols)
        for _ in range(self.maxlen):
            self.df = self.df.append(record, ignore_index=True)
        self.counter = 0
        self.humans = {}
        self.humans_counts = {}

    
    def PredictVelocity(self,msg):
        
        # df = self.df.copy()
        human_ids = []
        human_valuse = []
        human_pred_id = [] 
        human_c = [] 

        for num, hm in enumerate(msg.human_list):
            human_ids.append(hm.id)

            points = []
            for point in hm.body_key_points_with_prob:
                points += [point.x]
                points += [point.y]
            if hm.id in self.humans:
                df_tmp = self.humans[hm.id]
                counts = self.humans_counts[hm.id]
            else:
                df_tmp = self.df.copy()
                counts = self.counter
            df_tmp = df_tmp.shift(-1)
            df_tmp.at[self.maxlen-1] = points
            human_valuse.append(df_tmp)
            counts += 1
            human_c.append(counts)
            if counts >= self.maxlen:
                human_pred_id.append(hm.id)

        self.humans = dict(zip(human_ids,human_valuse))
        self.humans_counts = dict(zip(human_ids,human_c))


        # model name
        if len(human_pred_id) >= 1:
            pred_velocity_list =  PredHumanVelocityList()
            pred_velocity_list.header = msg.header
            for pred_id in human_pred_id:
                df_ = self.humans[pred_id]
                # #標準化する
                df_std = make_std(df_)
                predicted = self.loaded_model.predict([df_std.values.flatten()])
                pred_velocity =  PredHumanVelocity()
                pred_velocity.velocity = predicted
                pred_velocity.id = pred_id
                pred_velocity_list.human_list.append(pred_velocity)
            pub_pred_vel.publish(pred_velocity_list)

if __name__ == '__main__':
    # stop_model()
    rospy.loginfo('initialization+')
    rospy.init_node('predict_velocity_ros_node')
    # AlphaposeList = AlphaPoseHumanList()
    pred_cb = PredictModel()
    subscriber = rospy.Subscriber("/alphapose",AlphaPoseHumanList,pred_cb.PredictVelocity,queue_size = 1)
    pub_pred_vel = rospy.Publisher('/pred_velocity',PredHumanVelocityList , queue_size=1)

    # pub_pred_vel = rospy.Publisher('/pred_velocity',StringStamped , queue_size=1)
    rospy.spin()
    # while not rospy.is_shutdown():
    #     try:
    #         # alphapose_list = AlphaPoseHumanList()

    #         alphapose_list.header.stamp = img_time
    #         alphapose_list.header.frame_id = FrameId
    #         pose = demo.process(image_)
    #         cb_pose(alphapose_list,pose,image_)
    #     except:
    #         pass
    #         # try:
    #         #     make_img(image_)
    #         # except:
    #         #     pass
            
    #     rt.sleep()





