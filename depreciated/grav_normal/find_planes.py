#!/usr/bin/env python

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import struct
import rospy
import numpy as np
from std_msgs.msg import Int32, Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Imu, CameraInfo, Image, PointCloud2, PointField

from depth_to_pcd import depth_to_pcd
from get_normal import get_normal
from gravity_correction import gravity_correction
from get_mask import get_mask
from hsv import hsv_img

import matplotlib.pyplot as plt

class FindPlaneNode:
    def __init__(self):
        self.rescale_depth = 1000
        self.dot_bound = 0.9
        self.correction_iteration = 5
        self.kernel_size = 21
        self.cluster_size = 11

        self.imu = None
        self.camera_intrinsic = None

        rospy.init_node('find_plane_node')

        self.imu_topic = rospy.get_param('imu_filtered_topic', 100)
        self.depth_img_topic = rospy.get_param('depth_img_topic', 30)
        self.depth_intrinsic_topic = rospy.get_param('depth_intrinsic_topic', 1)
        self.pointcloud_topic = rospy.get_param('pointcloud_topic', 30)

        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imuCallback)
        self.depth_img_sub = rospy.Subscriber(self.depth_img_topic, Image, self.depthImageCallback)
        self.depth_intrinsic_sub = rospy.Subscriber(self.depth_intrinsic_topic, CameraInfo, self.depthIntrinsicCallback)

        self.pcd_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)
    def imuCallback(self, msg):
        self.imu = msg
    
    def depthIntrinsicCallback(self, msg):
        self.camera_intrinsic = msg
    
    def depthImageCallback(self, msg):
        downsample = 4
        downsample_root = int(downsample**0.5)

        if self.imu == None or self.camera_intrinsic == None:
            return False

        grav_normal = np.array([self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z])
        grav_normal = grav_normal / (np.linalg.norm(grav_normal) + 1e-15)

        # Find img normal
        img_normal = get_normal(depth.reshape(H,W), self.camera_intrinsic.K)
        if downsample!=0:
            img_normal = img_normal[::downsample_root,::downsample_root]

        img_normal_pos = img_normal.reshape(-1, 3)
        img_normal_neg = -img_normal_pos
        dot1 = np.dot(img_normal_pos, grav_normal).reshape(-1, 1)
        dot2 = np.dot(img_normal_neg, grav_normal).reshape(-1, 1)
        
        dot = np.concatenate((dot1, dot2), axis=1)
        normal_index = np.argmax(dot, axis=1)
        img_normal = np.zeros_like(img_normal_pos)
        img_normal[normal_index == 0] = img_normal_pos[normal_index == 0]
        img_normal[normal_index == 1] = img_normal_neg[normal_index == 1]

        if False:
            img_normal_rgb = (img_normal + 1)/2 * 255
            img_normal_rgb = img_normal_rgb.astype(np.uint8)
            img_normal_rgb = img_normal_rgb.reshape(H, W, 3)
            plt.imsave("/catkin_ws/src/gravity_plane_est/normal.png", img_normal_rgb)

        grav_normal = gravity_correction(grav_normal,img_normal, pts_3d, self.dot_bound,self.correction_iteration)

        if True:
            dot1 = np.dot(img_normal, grav_normal).reshape(-1,1)
            dot2 = np.dot(img_normal, -grav_normal).reshape(-1,1)

            angle_dist = np.concatenate((dot1, dot2), axis=1)
            angle_dist = np.max(angle_dist, axis=1)
            scalar_dist = np.dot(pts_3d, grav_normal)
            scalar_dist[angle_dist < self.dot_bound] = 0
            scalar_dist[pts_3d[:, 2] == 0] = 0

            # Plot histogram
            fig, ax = plt.subplots()
            ax.hist(scalar_dist[scalar_dist!=0], bins=1000)
            plt.xlabel("Distance")
            plt.ylabel("Count")
            plt.title("Histogram of Distance")
            fig.savefig("/catkin_ws/src/gravity_plane_est/histogram.png")

        
        mask = get_mask(grav_normal, img_normal, pts_3d, self.dot_bound, self.kernel_size, self.cluster_size)

        if False:
            plt.imsave("/catkin_ws/src/gravity_plane_est/mask.png", hsv_img(mask.reshape(H,W)))

        # Publish pointcloud

        if downsample!=0:
            H//=downsample_root
            W//=downsample_root

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        mask_color = hsv_img(mask.reshape(H,W)).reshape(-1,3)
        pcd_points = []
        for i in range(len(pts_3d)):
            x = pts_3d[i,0]
            y = pts_3d[i,1]
            z = pts_3d[i,2]

            r = mask_color[i,0]
            g = mask_color[i,1]
            b = mask_color[i,2]
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]

            pt = [x, y, z, rgb]

            pcd_points.append(pt)

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]

        pcd = point_cloud2.create_cloud(header, fields, pcd_points)

        self.pcd_pub.publish(pcd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = FindPlaneNode()
    node.run()