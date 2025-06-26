#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <vector>

ros::Publisher imu_msg_pub;
sensor_msgs::Imu imu_msg;
float alpha;
std::vector<sensor_msgs::Imu> store_imu_msg;
int store_imu_msg_pt = 0;
int store_imu_msg_max;


void imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    imu_msg.angular_velocity.x += - alpha * store_imu_msg[store_imu_msg_pt].angular_velocity.x + alpha * msg->angular_velocity.x;
    imu_msg.angular_velocity.y += - alpha * store_imu_msg[store_imu_msg_pt].angular_velocity.y + alpha * msg->angular_velocity.y;
    imu_msg.angular_velocity.z += - alpha * store_imu_msg[store_imu_msg_pt].angular_velocity.z + alpha * msg->angular_velocity.z;
    imu_msg.linear_acceleration.x += - alpha * store_imu_msg[store_imu_msg_pt].linear_acceleration.x + alpha * msg->linear_acceleration.x;
    imu_msg.linear_acceleration.y += - alpha * store_imu_msg[store_imu_msg_pt].linear_acceleration.y + alpha * msg->linear_acceleration.y;
    imu_msg.linear_acceleration.z += - alpha * store_imu_msg[store_imu_msg_pt].linear_acceleration.z + alpha * msg->linear_acceleration.z;
    imu_msg.orientation.x += - alpha * store_imu_msg[store_imu_msg_pt].orientation.x + alpha * msg->orientation.x;
    imu_msg.orientation.y += - alpha * store_imu_msg[store_imu_msg_pt].orientation.y + alpha * msg->orientation.y;
    imu_msg.orientation.z += - alpha * store_imu_msg[store_imu_msg_pt].orientation.z + alpha * msg->orientation.z;
    imu_msg.orientation.w += - alpha * store_imu_msg[store_imu_msg_pt].orientation.w + alpha * msg->orientation.w;
    
    store_imu_msg[store_imu_msg_pt] = *msg;
    store_imu_msg_pt++;
    store_imu_msg_pt%=store_imu_msg_max;

    // Publish the point cloud
    imu_msg_pub.publish(imu_msg);
}

int main(int argc, char** argv)
{    
    imu_msg.angular_velocity.x = 0;
    imu_msg.angular_velocity.y = 0;
    imu_msg.angular_velocity.z = 0;
    imu_msg.linear_acceleration.x = 0;
    imu_msg.linear_acceleration.y = 0;
    imu_msg.linear_acceleration.z = 0;
    imu_msg.orientation.x = 0;
    imu_msg.orientation.y = 0;
    imu_msg.orientation.z = 0;
    imu_msg.orientation.w = 0;

    // Initialize ROS
    ros::init(argc, argv, "avg_imu");

    // Create a node handle
    ros::NodeHandle nh;

    std::string imu_topic, imu_filtered_topic;
    nh.getParam("avg_imu/imu_topic", imu_topic);
    nh.getParam("avg_imu/imu_filtered_topic", imu_filtered_topic);
    nh.getParam("avg_imu/max_store", store_imu_msg_max);
    alpha = 1/(float)store_imu_msg_max;

    store_imu_msg = std::vector<sensor_msgs::Imu>(store_imu_msg_max);
    for(int i=0; i<store_imu_msg_max; i++){
        store_imu_msg[i].angular_velocity.x=0;
        store_imu_msg[i].angular_velocity.y=0;
        store_imu_msg[i].angular_velocity.z=0;
        store_imu_msg[i].linear_acceleration.x=0;
        store_imu_msg[i].linear_acceleration.y=0;
        store_imu_msg[i].linear_acceleration.z=0;
        store_imu_msg[i].orientation.x=0;
        store_imu_msg[i].orientation.y=0;
        store_imu_msg[i].orientation.z=0;
        store_imu_msg[i].orientation.w=0;
    }

    imu_msg_pub = nh.advertise<sensor_msgs::Imu>(imu_filtered_topic, 100);

    ros::Subscriber camera_info_sub = nh.subscribe(imu_topic, 100, imuCallback);

    // Spin to keep the node alive
    ros::spin();

    return 0;
}