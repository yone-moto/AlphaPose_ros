#include <ros/ros.h>

// #include <sensor_msgs/LaserScan.h>
#include <leg_tracker/Person.h>
#include <leg_tracker/PersonArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <alphapose_ros/AlphaPoseHumanList.h>

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

#include <numeric>
#include <tf/transform_listener.h>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>
template <typename T_p>
class SensorFusion
{
private:
    ros::NodeHandle nh;

    typedef message_filters::sync_policies::ApproximateTime<alphapose_ros::AlphaPoseHumanList, sensor_msgs::PointCloud2> sensor_fusion_sync_subs;

    message_filters::Subscriber<alphapose_ros::AlphaPoseHumanList> person_position_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub;
    message_filters::Synchronizer<sensor_fusion_sync_subs> sensor_fusion_sync;
    ros::Publisher pub_PersonArray;

    ros::Publisher pub_cloud;
    ros::Publisher pub_cloud_tf;
    ros::Publisher pub_hm_position;
    ros::Publisher pub_now_vel;

    tf::TransformListener listener;
    tf::StampedTransform transform;
    // tf::TransformListener transform_listener_;
    bool flag;
    const int cam_angle_w = 120; //realsense rgb angle realsenseのtpic infoみたいなのにのっているかも
    const int cam_angle_h = 120; //realsense rgb angle
    const int h = 480;
    const int w = 640;

    double theta_th_x_limit;
    double theta_th_y_limit;
    leg_tracker::PersonArray old_person_array;
    float is_tracker_ready_ = false;

public:
    SensorFusion();
    void Callback(const alphapose_ros::AlphaPoseHumanList::ConstPtr &, const sensor_msgs::PointCloud2::ConstPtr &);
    bool tflistener(std::string target_frame, std::string source_frame);
    void sensor_fusion(const alphapose_ros::AlphaPoseHumanList::ConstPtr, const sensor_msgs::PointCloud2::ConstPtr);
};

template <typename T_p>
SensorFusion<T_p>::SensorFusion()
    : nh("~"),
      //   topic name
      person_position_sub(nh, "/alphapose", 10), lidar_sub(nh, "/lidar", 10),
      sensor_fusion_sync(sensor_fusion_sync_subs(10), person_position_sub, lidar_sub)
{
    sensor_fusion_sync.registerCallback(boost::bind(&SensorFusion::Callback, this, _1, _2));
    pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/colord_cloud", 10);
    pub_cloud_tf = nh.advertise<sensor_msgs::PointCloud2>("/colord_cloud_tf", 10);
    pub_PersonArray = nh.advertise<leg_tracker::PersonArray>("/hm_detector", 10);
    pub_hm_position = nh.advertise<visualization_msgs::Marker>("/hm_position", 10);
    pub_now_vel = nh.advertise<std_msgs::Float32>("/now_velocity", 10);
    flag = false;
    nh.param("theta_th_x_limit", theta_th_x_limit, 8.0);
    nh.param("theta_th_y_limit", theta_th_y_limit, 8.0);
}

template <typename T_p>
void SensorFusion<T_p>::Callback(const alphapose_ros::AlphaPoseHumanList::ConstPtr &Person_pose,
                                 const sensor_msgs::PointCloud2::ConstPtr &pc2)
{
    if (!flag)
        tflistener(Person_pose->header.frame_id, pc2->header.frame_id);
    sensor_fusion(Person_pose, pc2);
}

// transformにtarget_frameの座標がsource_frame座標系で格納される
template <typename T_p>
bool SensorFusion<T_p>::tflistener(std::string target_frame, std::string source_frame)
{
    ros::Time time = ros::Time(0);

    try
    {
        listener.waitForTransform(target_frame, source_frame, time, ros::Duration(4.0));
        listener.lookupTransform(target_frame, source_frame, time, transform);
        return true;
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        ros::Duration(4.0).sleep();
        return false;
    }
}

template <typename T_p>
void SensorFusion<T_p>::sensor_fusion(const alphapose_ros::AlphaPoseHumanList::ConstPtr Person_pose,
                                      const sensor_msgs::PointCloud2::ConstPtr pc2)
{
    // std::cout << "ok1" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*pc2, *velodyne_cloud);

    typename pcl::PointCloud<T_p>::Ptr cloud(new pcl::PointCloud<T_p>);
    pcl::copyPointCloud(*velodyne_cloud, *cloud);

    // transform pointcloud from lidar_frame to camera_frame
    tf::Transform tf_pcl;
    tf_pcl.setOrigin(transform.getOrigin());
    tf_pcl.setRotation(transform.getRotation());
    typename pcl::PointCloud<T_p>::Ptr trans_cloud(new pcl::PointCloud<T_p>);
    pcl_ros::transformPointCloud(*cloud, *trans_cloud, tf_pcl);

    // SensorFusion Step
    // int s_human = ;
    std::vector<float> id_num;
    leg_tracker::PersonArray person_array;
    person_array.header = Person_pose->header;

    for (int hm = 0; hm < Person_pose->human_list.size(); hm++)
    {
        id_num.push_back(Person_pose->human_list[hm].id);
        float x_min = Person_pose->human_list[hm].body_bounding_box.x;
        float x_max = Person_pose->human_list[hm].body_bounding_box.x + Person_pose->human_list[0].body_bounding_box.width;
        float y_min = Person_pose->human_list[hm].body_bounding_box.y;
        float y_max = Person_pose->human_list[hm].body_bounding_box.y + Person_pose->human_list[0].body_bounding_box.height;

        float theta_x = -(cam_angle_w * (x_min + x_max) / 2.0 / w - (cam_angle_w / 2.0));
        float theta_y = -(cam_angle_h * (y_min + y_max) / 2.0 / h - (cam_angle_h / 2.0));
        float theta_x_th1 = (theta_x + theta_th_x_limit) * M_PI / 180;
        float theta_x_th2 = (theta_x - theta_th_x_limit) * M_PI / 180;
        float theta_y_th1 = (theta_y + theta_th_y_limit) * M_PI / 180;
        float theta_y_th2 = (theta_y - theta_th_y_limit) * M_PI / 180;

        std::vector<float> v;
        int pub_ = 0;

        for (typename pcl::PointCloud<T_p>::iterator pt = trans_cloud->points.begin(); pt < trans_cloud->points.end(); pt++)
        {
            float theta_x = atan2f((*pt).z, (*pt).x);
            float theta_y = atan2f((*pt).y, (*pt).x);

            // float range = sqrt( pow((*pt).x, 2.0) + pow((*pt).y, 2.0) + pow((*pt).z, 2.0));
            float range = sqrt(pow((*pt).x, 2.0) + pow((*pt).z, 2.0)); //もしかしたら違うかも
            if (theta_x_th1 >= theta_x && theta_x >= theta_x_th2 &&
                theta_y_th1 >= theta_y && theta_y >= theta_y_th2)
            {
                (*pt).b = 0;
                (*pt).g = 0;
                (*pt).r = 255;
                v.push_back(range);
                pub_ += 1;

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
            else
            {
                // std::cout << "ok3" << std::endl;
                (*pt).b = 0;
                (*pt).g = 0;
                (*pt).r += 0;
            }
        }
        if (pub_ >= 1)
        {

            std::vector<float>::iterator b = v.begin();
            std::vector<float>::iterator e = v.end();

            std::vector<float>::iterator q1 = b;
            std::vector<float>::iterator q2 = b;

            std::advance(q1, v.size() / 10); //適当　10分の1の値
            std::advance(q2, v.size() / 20); //適当　20分の1の値
            // This makes the 2nd position hold the median.
            std::nth_element(b, q1, e);
            std::nth_element(b, q2, e);

            float avg = (q1[2] + q2[2]) / 2.0;

            float x_ = avg * cosf(theta_x * M_PI / 180);
            float z_ = avg * sinf(theta_x * M_PI / 180);

            geometry_msgs::PoseStamped pose_new_person;
            geometry_msgs::PoseStamped pose_new_person_tf;
            pose_new_person.header = Person_pose->header;
            pose_new_person.pose.position.x = x_;
            pose_new_person.pose.position.z = z_;
            pose_new_person.pose.orientation.w = 1.0;

            // transform pointcloud from lidar_frame to camera_frame
            bool map_ = true;
            std::string frame_id_;

            if (map_)
            {
                ros::Time now = pc2->header.stamp;
                frame_id_ = "/map";
                listener.waitForTransform(frame_id_, Person_pose->header.frame_id, now, ros::Duration(5.0));
                listener.transformPose(frame_id_, pc2->header.stamp, pose_new_person, Person_pose->header.frame_id, pose_new_person_tf);
            }
            else
            {
                listener.transformPose(pc2->header.frame_id, pc2->header.stamp, pose_new_person, Person_pose->header.frame_id, pose_new_person_tf);
                frame_id_ = pc2->header.frame_id;
            }
            // visualize
            visualization_msgs::Marker m;
            m.header.frame_id = frame_id_;
            m.header.stamp = pc2->header.stamp;

            m.ns = "YONEMOTO";
            m.id = Person_pose->human_list[hm].id;
            m.type = visualization_msgs::Marker::SPHERE;
            m.action = visualization_msgs::Marker::ADD;
            m.pose = pose_new_person_tf.pose;
            m.scale.x = 0.1;
            m.scale.y = 0.5;
            m.scale.z = 0.1;
            m.color.a = 1.0; // Don't forget to set the alpha!
            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 0.0;
            pub_hm_position.publish(m);

            leg_tracker::Person NewPerson;
            NewPerson.pose = pose_new_person_tf.pose;
            NewPerson.id = Person_pose->human_list[hm].id;
            if (is_tracker_ready_)
            {
                for (int j = 0; j < old_person_array.people.size(); j++)
                {
                    if (NewPerson.id == old_person_array.people[j].id)
                    {
                        float diff_time = Person_pose->header.stamp.sec + Person_pose->header.stamp.nsec * 1e-9 - old_person_array.header.stamp.sec - old_person_array.header.stamp.nsec * 1e-9;
                        NewPerson.scale.x = (NewPerson.pose.position.x - old_person_array.people[j].pose.position.x) / diff_time;
                        NewPerson.scale.y = (NewPerson.pose.position.y - old_person_array.people[j].pose.position.y) / diff_time;
                        std_msgs::Float32 vel;
                        vel.data = sqrt(NewPerson.scale.x * NewPerson.scale.x + NewPerson.scale.y * NewPerson.scale.y);
                        pub_now_vel.publish(vel);
                    }
                }
            }
            person_array.people.push_back(NewPerson);
        }
    }
    old_person_array = person_array;
    is_tracker_ready_ = true;
    pub_PersonArray.publish(person_array);

    // transform pointcloud from camera_frame to lidar_frame
    typename pcl::PointCloud<T_p>::Ptr output_cloud(new pcl::PointCloud<T_p>);
    pcl_ros::transformPointCloud(*trans_cloud, *output_cloud, tf_pcl.inverse());

    sensor_msgs::PointCloud2 output_pc;
    pcl::toROSMsg(*output_cloud, output_pc);

    output_pc.header.frame_id = pc2->header.frame_id;
    output_pc.header.stamp = pc2->header.stamp;
    pub_cloud.publish(output_pc);

    sensor_msgs::PointCloud2 output_pc2;
    pcl::toROSMsg(*trans_cloud, output_pc2);

    output_pc2.header.frame_id = pc2->header.frame_id;
    output_pc2.header.stamp = pc2->header.stamp;
    pub_cloud_tf.publish(output_pc2);

    // Clear remaining markers in Rviz
    // for (int id_num_diff = num_prev_markers_published_- hm; id_num_diff > 0; id_num_diff--)
    // {
    //     // Clear remaining markers in Rviz
    //     visualization_msgs::Marker m;
    //     m.header.stamp = Person_pose->header.stamp;
    //     m.header.frame_id = Person_pose->header.frame_id;
    //     m.ns = "YONEMOTO";
    //     m.id = id_num_diff + hm;
    //     m.action = m.DELETE;
    //     pub_hm_position.publish(m);
    // }
    // num_prev_markers_published_ = hm; // For the next callback
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ucam_velodyn_fusion");
    ROS_INFO("person position");
    SensorFusion<pcl::PointXYZRGB> cr;
    ros::spin();
    return 0;
}
