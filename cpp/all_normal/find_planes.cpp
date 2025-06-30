#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <vector>
#include <yaml-cpp/yaml.h>

#include "../utils/data_conversion.cpp"
#include "../utils/math_utils.cpp"
#include "../utils/normal.cpp"
#include "../utils/visualise.cpp"
#include "../utils/correct_vector.cpp"
#include "../utils/find_peaks.cpp"
#include "cluster_normal.cpp"

#define VISUALISE true

YAML::Node config;
sensor_msgs::CameraInfo camera_info;
ros::Publisher cloud_pub;

bool cameraInfoReceived = false;

void depthIntrinsicCallback(const sensor_msgs::CameraInfo::ConstPtr& msg){
    camera_info.header = msg->header;
    camera_info.width = msg->width;
    camera_info.height = msg->height;
    camera_info.distortion_model = msg->distortion_model;
    camera_info.D = msg->D;
    camera_info.K = msg->K;
    camera_info.R = msg->R;
    camera_info.P = msg->P;

    cameraInfoReceived = true;
}

void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg){

    if (!cameraInfoReceived) {
        ROS_WARN("Camera Info not received yet.");
        return;
    }

    int W = msg->width;
    int H = msg->height;

    pcl::PointCloud<pcl::PointXYZRGB> points = DepthMsgToPointCloud(msg, camera_info, config["rescale_depth"].as<float>());

    std::vector< std::array<float, 3> > img_normals = get_normal(points);

    std::array< float, 3> z_axis = {0.0f, 0.0f, 1.0f};
    centre_hemisphere(img_normals, z_axis);

    std::vector< std::pair<int, std::array<float, 3> > > cardinal_directions = cluster_normal(img_normals, config["angle_bins"].as<int>(), config["angle_kernel_size"].as<int>());
}

int main(int argc, char** argv)
{    
    // Initialize ROS
    ros::init(argc, argv, "find_planes");

    // Create a node handle
    ros::NodeHandle nh;

    std::string depth_img_topic, depth_intrinsic_topic;
    nh.getParam("all_normal/depth_img_topic", depth_img_topic);
    nh.getParam("all_normal/depth_intrinsic_topic", depth_intrinsic_topic);

    std::string yaml_file_path;
    nh.getParam("all_normal/yaml_file_path", yaml_file_path);
    std::cout << "YAML file path: " << yaml_file_path << std::endl;
    config = YAML::LoadFile(yaml_file_path);

    ros::Subscriber depth_img_sub = nh.subscribe(depth_img_topic, 24, depthImageCallback);
    ros::Subscriber camera_info_sub = nh.subscribe(depth_intrinsic_topic, 1, depthIntrinsicCallback);

    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/all_normals_pcd", 1);

    // Spin to keep the node alive
    ros::spin();

    return 0;
}