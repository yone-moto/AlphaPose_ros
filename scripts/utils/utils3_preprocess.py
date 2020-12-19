#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy.stats as sp
import scipy
import warnings
warnings.simplefilter('ignore')
import glob
from scipy.spatial.transform import Rotation

# path の確認
import sys
import sys
sys.path.append('/home/robog/person_pose/utils')
try:
    print("remove path")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
from sklearn.metrics import r2_score,mean_squared_error
import lightgbm as lgb
from utils3 import make_dataset_df_to_ndarray,make_std
import pickle
import pickle


class depth_calculate():
    def __init__(self,x_length,y_length,point_list = [10,11,13,14],depth_path =  "/home/robog/bagfile/calibration/depth2"):
        self.x_length = x_length
        self.y_length = y_length
        self.max_range = 15
        self.th = 0.5
        self.depth_path = depth_base_path
        self.point_list = point_list
        
    def cal_distance(self,img,bgr):
        # velodyne の性能から
        tmp = (img.T[bgr].flatten()) / 255  * max_range 
        tmp = tmp[tmp > self.th]

        # ノイズを除去する
#         tmp_q1 = stats.scoreatpercentile(tmp, 1) #10パーセンと
        tmp_q3 = stats.scoreatpercentile(tmp, 70)
#         tmp = tmp[tmp > tmp_q1]
        tmp = tmp[tmp < tmp_q3]
        return list(tmp)
    
    def cal_theta(self,img,bgr):
        # velodyne の性能から
        tmp = (img.T[bgr].flatten()) / 255  * math.pi

        # ノイズを除去する
        tmp_q1 = stats.scoreatpercentile(tmp, 5) #10パーセンと
        tmp_q3 = stats.scoreatpercentile(tmp, 95)
        tmp = tmp[tmp > tmp_q1]
        tmp = tmp[tmp < tmp_q3]
        return tmp
    
    def cal_bbox(self,data,img,point_num = 10):
        
        if point_num == "center":
            bbox = 5
        else:
            bbox = 0
        min_x = int(data[f"x_{point_num}"]) - self.x_length 
        max_x = int(data[f"x_{point_num}"]) + self.x_length
        min_y = int(data[f"y_{point_num}"]) - self.y_length - bbox
        max_y = int(data[f"y_{point_num}"]) + self.y_length + bbox
        
        img = img[min_y:max_y,min_x:max_x]
        r = self.cal_distance(img,0)
        theta = self.cal_theta(img,1)
        
        return r
    
    def cal_path(self,data):

        depth_path = self.depth_path + "/" + data["image_id"].split(".")[0] + ".png"
        img = cv2.imread(depth_path)
        
                    
        # cal distance points
        for i,point in enumerate(self.point_list):
            dis = self.cal_bbox(data,img,point)
            if i == 0:
                depth = dis
            else:
                depth += dis
                
        # remove nan
        depth = np.array(depth)
        depth = depth[~np.isnan(depth)]
        depth = depth.flatten()
        
#         depth = [x for x in depth if str(x) != 'nan']
        
        # remove max min of 4 points up
#         if len(depth) >= 4:
#             depth.remove(max(depth))
#             depth.remove(min(depth))

        # ノイズを除去する
        depth = depth[depth > self.th]
        q1 = stats.scoreatpercentile(depth, 10) #10パーセンと
        q3 = stats.scoreatpercentile(depth, 75)
#         depth = depth[depth > q1]
        depth = depth[depth < q3]
        distance = np.mean(depth)
        return distance
    
def spilit_df(df_angle,judge_list):
    df_angle["judge"] = df_angle[judge_list].diff()
    not_conti_frame = df_angle[df_angle["judge"] != 1]#前後とのframeの差が２以上のものは離散的と判断
    start_list = not_conti_frame[:-1].index
    stop_list = not_conti_frame[1:].index
    return start_list,stop_list


def add_frame(df_pose,input_op_img_path):
    files_name = []
    files = glob.glob(input_op_img_path)
    for file in files:
        file = file.split("/")[-1]
        files_name.append(file)
    files_name.sort()
    df = pd.DataFrame(files_name,columns=["image_id"])
    df_tmp = df.merge(df_pose, how="left",on="image_id")
    df_tmp = df_tmp.reset_index()
    df_tmp["frame"] = df_tmp["index"]
    df_tmp.drop(columns=["index"],inplace=True)
    return df_tmp

def cal_vel_accel(df_tf,x = "field.transforms0.transform.translation.x",y = "field.transforms0.transform.translation.y",time = "%time"):
    vel_max = 2.0
    df_tf["time"] = df_tf[time]/10**(9) #sに変換
    df_tf['time_diff'] = df_tf['time'].diff() #差分を計算
    df_tf["dis_x"] = df_tf[x].diff() 
    df_tf["dis_y"] = df_tf[y].diff() 

    df_tf["dis"] = np.sqrt((df_tf["dis_x"].values**2 + df_tf["dis_y"].values**2))
    df_tf["vel"] =  df_tf["dis"]/df_tf["time_diff"] #速度の算出x,yのベクトルの大きさを時間で割る
    df_tf["accel"] =  df_tf["vel"]/df_tf["time_diff"] #速度の算出x,yのベクトルの大きさを時間で割る

    df_tf.loc[df_tf['vel'] > vel_max, 'vel'] = vel_max
    df_tf["vel_avg"] =  df_tf["vel"].rolling(100, center=True).sum()/100 #移動平均を算出　センサの値にのノイズがあるので

    df_tf["vel_x"] =  df_tf["dis_x"]/df_tf["time_diff"] #速度の算出xのベクトルの大きさを時間で割る
    df_tf["vel_y"] =  df_tf["dis_y"]/df_tf["time_diff"] #速度の算出yのベクトルの大きさを時間で割る
    df_tf["vel_x_avg"] =  df_tf["vel_x"].rolling(100, center=True).sum()/100 #移動平均を算出
    df_tf["vel_y_avg"] =  df_tf["vel_y"].rolling(100, center=True).sum()/100 #移動平均を算出
    return df_tf

def cal_yaw(df_tf,x = "field.transforms0.transform.rotation.x",y = "field.transforms0.transform.rotation.y",
            z = "field.transforms0.transform.rotation.z",w = "field.transforms0.transform.rotation.w"):
    # クォータニオンから
    # - x, y, z, w の順
    df_x = df_tf[x]
    df_y = df_tf[y]
    df_z = df_tf[z]
    df_w = df_tf[w]
    df_x_avg =  df_x.rolling(100, center=True).sum().fillna(1)/100
    df_y_avg =  df_y.rolling(100, center=True).sum().fillna(1)/100
    df_z_avg =  df_z.rolling(100, center=True).sum().fillna(1)/100
    df_w_avg =  df_w.rolling(100, center=True).sum().fillna(1)/100

    quat = np.array([df_x,df_y,df_z,df_w])
    quat_avg = np.array([df_x_avg,df_y_avg,df_z_avg,df_w_avg])

    rot = Rotation.from_quat(quat.T)
    rot_avg = Rotation.from_quat(quat_avg.T)

    tmp = rot.as_euler('xyx')
    tmp_avg = rot_avg.as_euler('xyx')

    df_tf["yaw"] = tmp[:,2]
    df_tf["yaw_avg"] = tmp_avg[:,2]
    return df_tf


# ベクトル回転関数
# deg=Falseならばラジアンで角度を指定
# deg=Trueならば度数単位で角度を指定

def rotation_o(df):
    df_temp = df.copy()
    t = -np.arctan2(df_temp["y_robot"],df_temp["x_robot"])
    u = (df_temp["x_person"],df_temp["y_person"])
    # 回転行列
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    df_temp["x_person_rot"],df_temp["y_person_rot"] = np.dot(R, u)
    return df_temp

def pred_angle(df):
    label_frame = 0
    #30frame分
    past_num = 1
    pose_mode = True
    lower_body = True
    upper_body = True
    size_mode = False
    vel_person = False
    
    label_list=["yaw"]
    
    filename = f'/home/robog/person_pose/model/lightgbm/angle_pose_{pose_mode}_{past_num}_{label_frame}_25.sav'
    
    train_list_x = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder"
                    ,"LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee",
                    "RKnee","LAnkle","RAnkle", "Head", "Neck", "Hip","LBigToe",
                    "RBigToe","LSmallToe","RSmallToe","LHeel","RHeel" ]
    train_list_y = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder"
                    ,"LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee",
                    "RKnee","LAnkle","RAnkle", "Head", "Neck", "Hip","LBigToe",
                    "RBigToe","LSmallToe","RSmallToe","LHeel","RHeel" ]
    
    #標準化する
    df_std = make_std(df,train_list_x,train_list_y,pose_mode=pose_mode,vel=vel_person)
    X = df_std.drop(columns=label_list)
    model = pickle.load(open(filename, 'rb'))
    predicted = model.predict(X)
    pre = pd.DataFrame(predicted)
    df["pred"] = predicted
    return df