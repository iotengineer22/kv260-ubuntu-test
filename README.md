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
