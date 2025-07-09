# Depth-Image Plane estimator

This repository includes two different plane detectors in ROS1. Both methods require depth image and depth camera intrinsics as inputs. The gravity normal plane estimator also requires an IMU as input. 

Two existing plane detectors are also implemented as benchmarks.

Points belonging to each detected plane are assigned a corresponding color. This is then publised as a point cloud topic which can be visualized in RViz.

## Installation
To build the package, you need to have a catkin workspace set up.
```
cd ~/catkin_ws/src
git clone https://github.com/tauzn-clock/depth_plane_est
cd ~/catkin_ws
catkin build
```

## Docker

Build the docker image using the following command:

```
docker build 
    --ssh default=$SSH_AUTH_SOCK 
    -t depth_plane_est .
```

Run the docker image using the following command:

```
docker run  
    -it 
    -v ~/scratchdata:/scratchdata 
    --gpus all 
    --shm-size 16g  
    -d  
    --network=host  
    --restart unless-stopped  
    --env="DISPLAY"  
    --env="QT_X11_NO_MITSHM=1"  
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   
    --device=/dev/ttyUSB0  
    -e DISPLAY=unix$DISPLAY  
    --privileged 
    depth_plane_est
```

## Run

Start the ROS master node with a running camera node or playback of a rosbag file. Then run any of the plane detectors.

## Plane Detectors
> All default parameters are set for Orbbec Gemini L camera. If you are using a different camera, you need to change the parameters accordingly.
### Gravity Normal Plane Estimator

```
roslaunch depth_plane_est gravity_normal.launch
```

Launchfile parameters:
- `imu_topic`: IMU data topic. Default is `/camera/gyro_accel/sample`.
- `imu_filtered_topic`: Filtered IMU data topic. Default is `/camera/gyro_accel/sample_filtered`.
- `max_store`: Number of IMU readings stored for averaging filter. Default is 50.
- `depth_img_topic`: Depth image topic. Default is `/camera/depth/image_raw`.
- `depth_intrinsic_topic`: Depth camera intrinsics topic. Default is `/camera/depth/camera_info`.
- `yaml_file`: YAML file containing parameters for the plane estimator. Default is `/catkin_ws/src/depth_plane_est/cpp/gravity_normal/gravity_normal.yaml`.

YAML file parameters:
- `rescale_depth`: Rescale factor for depth readings. Default is 0.001.
- `dot_bound`: Dot product threshold when selecting normal vectors used for gravity vector correction. Default is 0.9.
- `correction_iteration`: Number of iterations for gravity vector correction. Default is 5.
- `kernel_size`: Size of the kernel used for peak detection when finding planes normal to gravity. Default is 21.
- `cluster_size`: Cluster size of points centred around each peak. Default is 5.
- `plane_ratio`: Minimum number of points as ratio of overall image size (H*W). Default is 0.01.

Output PCD Topic: `/grav_normals_pcd`

Output PCD Frame: `camera_link`

### All Normal Plane Estimator

```
roslaunch depth_plane_est all_normal.launch
```

Launchfile parameters:
- `depth_img_topic`: Depth image topic. Default is `/camera/depth/image_raw`.
- `depth_intrinsic_topic`: Depth camera intrinsics topic. Default is `/camera/depth/camera_info`.
- `yaml_file`: YAML file containing parameters for the plane estimator. Default is `/catkin_ws/src/depth_plane_est/cpp/all_normal/all_normal.yaml`.

YAML file parameters:
- `rescale_depth`: Rescale factor for depth readings. Default is 0.001.
- `angle_bins`: Number of angle bins used to discretise angles around the x and y axis. Default is 41.
- `angle_kernel_size`: Size of the kernel used for peak detection when finding cardinal vectors. Default is 5.
- `directions_selected`: Number of cardinal vectors selected. Default is 3.
- `dot_bound`: Dot product threshold when selecting normal vectors used for each cardinal vector correction. Default is 0.9.
- `correction_iteration`: Number of iterations for gravity vector correction. Default is 5.
- `kernel_size`: Size of the kernel used for peak detection when finding planes normal to each cardinal vector. Default is 21.
- `cluster_size`: Cluster size of points centred around each peak. Default is 5.
- `plane_ratio`: Minimum number of points as ratio of overall image size (H*W). Default is 0.001.

Output PCD Topic: `/all_normals_pcd`

Output PCD Frame: `camera_link`

### Organised Plane Estimator

```
roslaunch depth_plane_est benchmark_organised.launch
```

Uses the Organised Multiplane Segmentation algorithm from PCL library.

Launchfile parameters:
- `yaml_file`: YAML file containing parameters for the plane estimator. Default is `/catkin_ws/src/depth_plane_est/cpp/organised_plane/organised.yaml`.