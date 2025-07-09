# Depth-Image Plane estimator

This repository includes two different plane detector implementation in ROS1. Both methods require depth image and depth camera intrinsics as inputs. The gravity normal plane estimator also requires an IMU as input. 

I have also implemented two existing plane detectors as a benchmark.

## Building Docker

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

