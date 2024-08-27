# kv260-ubuntu-test
This repository present solution for my KV260 Ubuntu test

## Jupyter_notebooks and Python Test with YOLOX

### Test and Synthesis Environment

First, let's explain the testing environment.

Compilation and synthesis are conducted on a separate desktop PC (Ubuntu) rather than the KV260.

The tools and synthesis environment are as follows:

- **Vivado, Vitis:** 2023.1
- **Vitis AI:** 3.5 → 2.5


---

### Using the Vitis AI YOLOX Model

I obtained the official Vitis AI model. Although the compilation was done with version 2.5, the YOLOX model for object detection was sourced from version 3.5.

It worked without issues. Download and extract the sample model as follows:

```bash
wget https://www.xilinx.com/bin/public/openDownload?filename=pt_yolox-nano_3.5.zip
unzip openDownload\?filename\=pt_yolox-nano_3.5.zip
```

---

### Setting Up the Vitis AI 2.5 Docker Environment

Setting up an older Vitis AI environment is simple.

Instead of pulling the latest Docker image, specify version "2.5". Check the available versions here:

[Docker Hub - Xilinx Vitis AI CPU](https://hub.docker.com/r/xilinx/vitis-ai-cpu/tags)

```bash
cd Vitis-AI/
docker pull xilinx/vitis-ai-cpu:2.5
```

After that, start Vitis AI and compile the model. For this test, I prepared both B512 and B4096 models, representing the smallest and largest sizes available for the KV260's DPU.

The command to run the Vitis AI Docker is as follows:

```bash
./docker_run.sh xilinx/vitis-ai-cpu:2.5
conda activate vitis-ai-pytorch
cd pt_yolox-nano_3.5/
vai_c_xir -x quantized/YOLOX_0_int.xmodel -a b512_arch.json -n b512_2_5_yolox_nano_pt -o ./b512_2_5_yolox_nano_pt
```

---

### Synthesis of the KV260 DPU (B512) Environment

For details on synthesizing the DPU environment, please refer to the following articles. They provide a comprehensive guide, including benchmarks for B512~B4096 on similar boards like the KR260.

- [Benchmark Architectures of the DPU with KR260](https://www.hackster.io/iotengineer22/benchmark-architectures-of-the-dpu-with-kr260-699f19)
- [Implementation of DPU, GPIO, and PWM for KR260](https://www.hackster.io/iotengineer22/implementation-dpu-gpio-and-pwm-for-kr260-f7637b)

---

### Testing YOLOX on KV260 with Jupyter Notebooks

The KV260's OS is Ubuntu, specifically Ubuntu Desktop 22.04 LTS.

Download it from [Ubuntu](https://ubuntu.com/download/amd) and write the image to an SD card to boot up Ubuntu on the KV260.

The evaluation board used is the KV260.

First, install PYNQ on the KV260:

```bash
sudo snap install xlnx-config --classic --channel=2.x
xlnx-config.sysinit
sudo poweroff
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ/
sudo bash install.sh -b KV260
```

The Jupyter Notebooks program is available here:

[KV260 Ubuntu Test - Jupyter Notebooks](https://github.com/iotengineer22/kv260-ubuntu-test/blob/main/jupyter_notebooks/pynq-yolox/dpu_yolox-nano_pt_coco2017.ipynb)

Test YOLOX using DPU-PYNQ via Jupyter Notebooks.

Copy the program to the Jupyter Notebooks folder as shown in the example below:

```bash
sudo su
cd $PYNQ_JUPYTER_NOTEBOOKS
cd jupyter_notebooks/
ls
cp -rf /home/ubuntu/kv260-ubuntu-test/jupyter_notebooks/pynq-yolox/ ./
```

Access the KV260 via a web browser and run the program. The DPU (dpu.bit) used is the default B4096 provided by DPU-PYNQ, so the YOLOX PyTorch model was also prepared for 4096.

In testing, five objects were successfully detected using YOLOX.

---

### Testing YOLOX with Python and KV260

Test YOLOX with DPU-PYNQ using Python.

The program, DPU, and YOLOX model are already prepared. They can be found here:

[KV260 YOLOX Test](https://github.com/iotengineer22/kv260-ubuntu-test/tree/main/yolox-test)

First, switch to the root user and start the PYNQ virtual environment. Then execute the program:

```bash
sudo su
cd kv260-ubuntu-test/
cd yolox-test/
source /etc/profile.d/pynq_venv.sh
python3 app_yolox_nano_pt_coco2017.py
```

The test results show that five objects were detected. Execution times, including pre-processing, post-processing, and inference time, were also recorded:

```
Details of detected objects: [67. 64. 32. 66. 64.]
Pre-processing time: 0.0078 seconds
DPU execution time: 0.0227 seconds
Post-process time: 0.0296 seconds
Total run time: 0.0601 seconds
Performance: 16.64670582632164 FPS
```


Here's the text translated and formatted for a GitHub README:

---

## KV260 ROS2 and GPIO Testing with a Webcam

In this project, I tested ROS2 and GPIO functionality on the KV260, integrating it with a webcam. The object detection data processed by the DPU-PYNQ on the KV260 was utilized for various applications.


### Webcam (Logitech C270n)

The webcam used is the Logitech C270n. Despite its low cost, it supports 640x480 resolution at 30fps, making it sufficient for testing purposes.


### Installing GStreamer, libuvc, and v4l2loopback-dkms

To enable live streaming from the USB camera, the following libraries need to be installed:

```bash
git clone https://github.com/nickel110/libuvc.git
cd libuvc/
mkdir build
cd build/
cmake ..
make && sudo make install

sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
sudo apt install v4l2loopback-dkms
```

Since GStreamer is not included in the default OpenCV-Python package on PYNQ, you'll need to uninstall it and link it to Ubuntu's OpenCV:

```bash
sudo su
source /etc/profile.d/pynq_venv.sh
pip uninstall opencv-python
```

### Installing ROS2 and rviz2

There are various ways to install ROS2 on the KV260. For convenience, I used the reference script from a previous AMD (Xilinx) FPGA contest.

[Installing ROS2-Humble](https://github.com/amd/Kria-RoboticsAI/blob/main/files/scripts/install_ros.sh)

```bash
sudo su
source /etc/profile.d/pynq_venv.sh
source ./install_ros.sh
```

Install visualization tools and ROS2 OpenCV packages:

```bash
sudo apt install ros-humble-rviz2
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-cv-bridge
```

### Testing YOLOX with GPIO Output Using a Webcam

I tested live streaming with a webcam while performing object detection using YOLOX. When an orange ball is detected, a GPIO output is triggered to blink an LED. The program used for this test is available [here](https://github.com/iotengineer22/kv260-ubuntu-test/blob/main/usb-camera/app_gst-yolox-real-normal-camera-gpio.py).

For GPIO output, I used a custom debug board that I made, which is documented [here](https://github.com/iotengineer22/PCB-KV260-PMOD-TEST).

Here’s how I tested it:

```bash
sudo su
source /etc/profile.d/pynq_venv.sh
cd kv260-ubuntu-test/usb-camera/
python3 app_gst-yolox-real-normal-camera-gpio.py
```

### Testing YOLOX with ROS2 Output Using a Webcam

I also tested live streaming with ROS2 output, where the detected objects and coordinates are displayed as markers. The program for this test is available [here](https://github.com/iotengineer22/kv260-ubuntu-test/blob/main/usb-camera/gst-yolox-ros2-normal-camera.py).

First, start rviz2:

```bash
sudo su
source /opt/ros/humble/setup.bash
rviz2
```

Display the "MarkerArray" and "Image" from the left window in rviz2:

- **Image:** `/camera/image/q1`
- **Fixed Frame:** `base_link`

In another terminal, run the following:

```bash
sudo su
source /etc/profile.d/pynq_venv.sh
source /opt/ros/humble/setup.bash
cd kv260-ubuntu-test/usb-camera/
python3 gst-yolox-ros2-normal-camera.py
```



## Connecting and Testing the RICOH THETA V 360° Camera with KV260

In this project, we are testing the connection of the RICOH THETA V 360° camera to the KV260 platform. We will also be checking the GStreamer debug logs while comparing performance between 2K and 4K images.

### Setup

To enable GStreamer support for the THETA camera, we installed the necessary libraries using the following steps:

```bash
git clone https://github.com/nickel110/gstthetauvc.git
cd gstthetauvc/thetauvc/
make
sudo find / -type d -name 'gstreamer-1.0'
ls /usr/lib/aarch64-linux-gnu/gstreamer-1.0
sudo cp gstthetauvc.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
ls /usr/lib/aarch64-linux-gnu/gstreamer-1.0
sudo /sbin/ldconfig -v
gst-inspect-1.0 thetauvcsrc
```

### Test Results

We conducted object detection tests with the 360° YOLOX model, as shown in the example below. 

While USB 2.0 theoretically supports up to 480Mbps, actual performance may be lower. During the contest, we did not use 4K (3840p) resolution, and found that 2K (1920p) was sufficiently satisfying.

```bash
sudo su
cd kv260-ubuntu-test/yolox-test/
source /etc/profile.d/pynq_venv.sh
export GST_DEBUG=4
python3 app_gst-yolox-real-360-2divide.py
```

- **2K Image**: The bitrate was approximately 40Mbps. Object detection in 360° was running at around 6fps.

```text
bitrate=(uint)43181526;

0:00:07.751492592  5122 0xaaaaf53132a0 INFO              GST_STATES gstbin.c:2069:gst_bin_get_state_func:<pipeline0> getting state
0:00:07.752487562  5122 0xaaaaf5fed360 INFO               baseparse gstbaseparse.c:4088:gst_base_parse_set_latency:<h264parse0> min/max latency 0:00:00.033366666, 0:00:00.033366666
0:00:07.752937767  5122 0xaaaaf5fed360 INFO            videodecoder gstvideodecoder.c:1633:gst_video_decoder_sink_event_default:<avdec_h264-0> upstream tags: taglist, video-codec=(string)"H.264\ \(Baseline\ Profile\)", minimum-bitrate=(uint)42394646, maximum-bitrate=(uint)43093787, bitrate=(uint)43181526;
```

- **4K Image**: The bitrate was approximately 120Mbps. Object detection in 360° was running at around 3fps.

```text
bitrate=(uint)125117561;

0:00:39.427957699  5325 0xaaaae1e9b2a0 INFO              GST_STATES gstbin.c:2069:gst_bin_get_state_func:<pipeline0> getting state
0:00:39.430027031  5325 0xaaaae3896f60 INFO               baseparse gstbaseparse.c:4088:gst_base_parse_set_latency:<h264parse0> min/max latency 0:00:00.033366666, 0:00:00.033366666
0:00:39.430708689  5325 0xaaaae3896f60 INFO            videodecoder gstvideodecoder.c:1633:gst_video_decoder_sink_event_default:<avdec_h264-0> upstream tags: taglist, video-codec=(string)"H.264\ \(Baseline\ Profile\)", minimum-bitrate=(uint)123949572, maximum-bitrate=(uint)125917524, bitrate=(uint)125117561;
```


Here’s the content translated into English and formatted for a GitHub README:

---

## Stepper Motor Operation Test

This repository provides examples of how to operate a stepper motor using the KV260.

We used the KV260 evaluation board for these tests, and the programming was done in Python.

The program and model files should be placed in the `jupyter_notebooks` folder as shown below:

```bash
sudo su
cd $PYNQ_JUPYTER_NOTEBOOKS
cd jupyter_notebooks
cp -rf /home/ubuntu/kv260-ubuntu-test/jupyter_notebooks/pynq-stepper/ ./
```

### Stepper Motor Control with Custom Driver Circuit

Next, we used a custom driver circuit to control the stepper motor. The driver IC used is the DRV8833, implemented on a custom PCB. The motor is powered by the USB 5V from the evaluation board.

We confirmed that the motor rotates correctly and the waveform is stable. We also tested the operation with a step frequency increased to 125Hz, and it worked without any issues.

### Stepper Motor and Object Detection Test

This is an advanced test where a webcam and stepper motor are connected to the KV260. The motor is controlled to rotate when an orange ball is detected by the webcam.

To start the Python program:

```bash
cd kv260-ubuntu-test/stepper-motor/
source /etc/profile.d/pynq_venv.sh
python3 app_gst-yolox-real-normal-camera-stepper.py
```

The actual test program is saved here:

[https://github.com/iotengineer22/kv260-ubuntu-test/blob/main/stepper-motor/app_gst-yolox-real-normal-camera-stepper.py](https://github.com/iotengineer22/kv260-ubuntu-test/blob/main/stepper-motor/app_gst-yolox-real-normal-camera-stepper.py)

When the orange ball is not detected by the webcam, the stepper motor does not rotate. The motor only rotates when the ball is detected.


## KV260 PWM Test with PYNQ

I tried modifying the AXI bus registers using PYNQ's MMIO.

```bash
sudo su
cd $PYNQ_JUPYTER_NOTEBOOKS
cd jupyter_notebooks
cp -rf /home/ubuntu/kv260-ubuntu-test/jupyter_notebooks/pynq-pwm/ ./
```




