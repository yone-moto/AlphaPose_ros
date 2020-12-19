#include <ros/ros.h>

// #include <sensor_msgs/LaserScan.h>

#include <sensor_msgs/PointCloud2.h>
#include<alphapose_ros/AlphaPoseHumanList.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include <cmath>
#include <bits/stdc++.h>

template<typename T_p>
class SensorFusion{
    private:
        ros::NodeHandle nh;

        typedef message_filters::sync_policies::ApproximateTime<alphapose_ros::AlphaPoseHumanList, sensor_msgs::PointCloud2> sensor_fusion_sync_subs;

        message_filters::Subscriber<alphapose_ros::AlphaPoseHumanList> person_position_sub;
        message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
        message_filters::Synchronizer<sensor_fusion_sync_subs> sensor_fusion_sync;

        ros::Publisher pub_cloud;
        ros::Publisher pub_cloud3;


        tf::TransformListener listener;
        tf::StampedTransform  transform;
        bool flag;
        double cam_angle_w = 120; //realsense rgb angle realsenseのtpic infoみたいなのにのっているかも
        double cam_angle_h = 120; //realsense rgb angle
        double h = 480;
        double w = 640;

        double theta_th1_limit ;
        double theta_th2_limit ;

    public:
        SensorFusion();
        void Callback(const alphapose_ros::AlphaPoseHumanList::ConstPtr&, const sensor_msgs::PointCloud2::ConstPtr&);
        bool tflistener(std::string target_frame, std::string source_frame);
        void sensor_fusion(const alphapose_ros::AlphaPoseHumanList::ConstPtr, const sensor_msgs::PointCloud2::ConstPtr);
};

template<typename T_p>
SensorFusion<T_p>::SensorFusion()
    : nh("~"),
    //   topic name 
    // r,thetaのmsgの型がどれが適切わからなかったのでLaserScanにした。独自につくっても構わない。そのときはheaderつきで
      person_position_sub(nh, "/alphapose", 10), lidar_sub(nh, "/lidar", 10),
      sensor_fusion_sync(sensor_fusion_sync_subs(10), person_position_sub, lidar_sub)
{
    sensor_fusion_sync.registerCallback(boost::bind(&SensorFusion::Callback, this, _1, _2));
    pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/colord_cloud", 10);
    pub_cloud3 = nh.advertise<sensor_msgs::PointCloud2>("/colord_cloud3", 10);
    flag = false;
    nh.param("theta_th1_limit", theta_th1_limit, 8.0);
    nh.param("theta_th2_limit", theta_th2_limit, 8.0);

}


template<typename T_p>
void SensorFusion<T_p>::Callback(const alphapose_ros::AlphaPoseHumanList::ConstPtr& Person_pose,
                             const sensor_msgs::PointCloud2::ConstPtr& pc2)
{
    if(!flag) tflistener(Person_pose->header.frame_id, pc2->header.frame_id);    
    sensor_fusion(Person_pose, pc2);

}

// transformにtarget_frameの座標がsource_frame座標系で格納される
template<typename T_p>
bool SensorFusion<T_p>::tflistener(std::string target_frame, std::string source_frame)
{
    ros::Time time = ros::Time(0);

    try{
        listener.waitForTransform(target_frame, source_frame, time, ros::Duration(4.0));
        listener.lookupTransform(target_frame, source_frame,  time, transform);
        return true;
    }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
        ros::Duration(4.0).sleep();
        return false;
    }
}

template<typename T_p>
void SensorFusion<T_p>::sensor_fusion(const alphapose_ros::AlphaPoseHumanList::ConstPtr Person_pose,
                             const sensor_msgs::PointCloud2::ConstPtr pc2)
{
    // std::cout << "ok1" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*pc2, *velodyne_cloud);
    
    typename pcl::PointCloud<T_p>::Ptr cloud(new pcl::PointCloud<T_p>);
    pcl::copyPointCloud(*velodyne_cloud, *cloud);

    // transform pointcloud from lidar_frame to camera_frame
    tf::Transform tf;
    tf.setOrigin(transform.getOrigin());
    tf.setRotation(transform.getRotation());
    typename pcl::PointCloud<T_p>::Ptr trans_cloud(new pcl::PointCloud<T_p>);
    pcl_ros::transformPointCloud(*cloud, *trans_cloud, tf);

    // SensorFusion Step
    // typename pcl::PointCloud<T_p>::Ptr colored_cloud(new pcl::PointCloud<T_p>);
    // *colored_cloud = *trans_cloud; 

    float x_min = Person_pose->human_list[0].body_bounding_box.x;
    float x_max = Person_pose->human_list[0].body_bounding_box.x + Person_pose->human_list[0].body_bounding_box.width;
    float y_min = Person_pose->human_list[0].body_bounding_box.y;
    float y_max = Person_pose->human_list[0].body_bounding_box.y + Person_pose->human_list[0].body_bounding_box.height;

    double theta_x = -(cam_angle_w * (x_min + x_max) / 2.0 /w  - (cam_angle_w/2.0));
    double theta_y = -(cam_angle_h * (y_min + y_max) / 2.0 /h  - (cam_angle_h/2.0));
    double theta_x_th1 = (theta_x  + theta_th1_limit) * M_PI/180;
    double theta_x_th2 = (theta_x  - theta_th1_limit) * M_PI/180;
    double theta_y_th1 = (theta_y  + theta_th2_limit + 4) * M_PI/180;
    double theta_y_th2 = (theta_y  - theta_th2_limit) * M_PI/180;

    for(typename pcl::PointCloud<T_p>::iterator pt=trans_cloud->points.begin(); pt<trans_cloud->points.end(); pt++)
    {
        float theta_x = atan2f((*pt).z,(*pt).x);
        float theta_y = atan2f((*pt).y,(*pt).x);

        // float range = sqrt( pow((*pt).x, 2.0) + pow((*pt).y, 2.0) + pow((*pt).z, 2.0)); 
        float range = sqrt( pow((*pt).x, 2.0) + pow((*pt).z, 2.0));//もしかしたら違うかも

        if( theta_x_th1 > theta_x && theta_x > theta_x_th2 && 
        theta_y_th1 > theta_y && theta_y > theta_y_th2){
            (*pt).b = 0;
            (*pt).g = 0;
            (*pt).r = 255;

        //     }
        // else if (theta_x_th1 > theta_x && theta_x > theta_x_th2 )
        // {
        //     (*pt).b = 0;
        //     (*pt).g = 255;
        //     (*pt).r = 0;
        // }

        // else if (theta_y_th1 > theta_y && theta_y > theta_y_th2 )
        // {
        //     (*pt).b = 255;
        //     (*pt).g = 255;
        //     (*pt).r = 0;
        }
        else{
            // std::cout << "ok3" << std::endl;
            (*pt).b = 255;
            (*pt).g = 255;
            (*pt).r = 255;
            }
        
    }

    // transform pointcloud from camera_frame to lidar_frame
    typename pcl::PointCloud<T_p>::Ptr output_cloud(new pcl::PointCloud<T_p>);
    pcl_ros::transformPointCloud(*trans_cloud, *output_cloud, tf.inverse());

    sensor_msgs::PointCloud2 output_pc2;
    pcl::toROSMsg(*trans_cloud, output_pc2);
    
    output_pc2.header.frame_id = pc2->header.frame_id;
    output_pc2.header.stamp = pc2->header.stamp;
    pub_cloud.publish(output_pc2);

    sensor_msgs::PointCloud2 output_pc3;
    pcl::toROSMsg(*output_cloud, output_pc3);
    
    output_pc3.header.frame_id = pc2->header.frame_id;
    output_pc3.header.stamp = pc2->header.stamp;
    pub_cloud3.publish(output_pc3);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ucam_velodyn_fusion");
    ROS_INFO("person position");
    SensorFusion<pcl::PointXYZRGB> cr;
    ros::spin();
    return 0;
}
