#!/usr/bin/env python3
import pandas as pd
from utils.utils3 import make_std
import numpy as np
import glob
import pickle
import rospy

from alphapose_ros.msg import AlphaPoseHumanList
from alphapose_ros.msg import AlphaPoseHuman

from std_msgs.msg import Float32
# from jsk_rviz_plugins.msg import StringStamped
class PredictModel():
    def __init__(self,filename = "/home/robog/catkin_ws/src/alphapose_ros/scripts/models/RosModel.sav"):
        self.maxlen = 10
        self.input_list = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder"
                        ,"LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee",
                        "RKnee","LAnkle","RAnkle", "Head", "Neck", "Hip","LBigToe",
                        "RBigToe","LSmallToe","RSmallToe","LHeel","RHeel"]
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
    
    def PredictVelocity(self,msg):
        
        # df = self.df.copy()

        for hm_id, hm in enumerate(msg.human_list):
            if hm_id==0:
                self.counter += 1 
                self.df = self.df.shift(-1)
                poins = []
                for point in hm.body_key_points_with_prob:
                    poins += [point.x]
                    poins += [point.y]
                self.df.at[self.maxlen-1] = poins
        # model name
        if self.counter >= self.maxlen:
            # #標準化する
            df_std = make_std(self.df)
            predicted = self.loaded_model.predict([df_std.values.flatten()])
            # pub_data = StringStamped()
            # pub_data.data = predicted
            # pub_data.header = msg.header
            pub_pred_vel.publish(predicted)

    
if __name__ == '__main__':
    # stop_model()
    rospy.loginfo('initialization+')
    rospy.init_node('predict_velocity_ros_node')
    # AlphaposeList = AlphaPoseHumanList()
    pred_cb = PredictModel()
    subscriber = rospy.Subscriber("/alphapose",AlphaPoseHumanList,pred_cb.PredictVelocity,queue_size = 1)
    pub_pred_vel = rospy.Publisher('/pred_velocity',Float32 , queue_size=1)

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





