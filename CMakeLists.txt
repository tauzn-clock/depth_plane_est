cmake_minimum_required(VERSION 3.0.2)
project(depth_plane_est)

# Find catkin and dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find the YAML-CPP library (for YAML support)
find_package(yaml-cpp REQUIRED)

# Declare a catkin package
catkin_package()

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(get_organised_pcd cpp/benchmark/get_organised_pcd.cpp)
target_link_libraries(get_organised_pcd
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_executable(get_ahc_pcd cpp/benchmark/get_ahc_pcd.cpp)
target_link_libraries(get_ahc_pcd 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

add_executable(avg_imu cpp/gravity_normal/avg_imu.cpp)
target_link_libraries(avg_imu
  ${catkin_LIBRARIES}
)

add_executable(gravity_normal cpp/gravity_normal/find_planes.cpp)
target_link_libraries(gravity_normal
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)