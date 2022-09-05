# [Hệ thống phần mềm AI nhận diện khuôn mặt VKIST - Máy chủ] 

 ![version](https://img.shields.io/badge/version-1.0.0-blue.svg) 

```bash
$ # Get the code
$ git clone https://github.com/vietbacnguyen96/AI_Award_2022.git
$ cd AI_Award_2022
$
$ # Get the weight of arcface and put it on folder /app/ms1mv3_arcface_r50_fp16
$ wget https://github.com/vietbacnguyen96/AI_Awards_Server/releases/download/v1.0.0/backbone_ir50_ms1m_epoch120.pth
$ mkdir \app\ms1mv3_arcface_r50_fp16
$ mv backbone_ir50_ms1m_epoch120.pth /app/ms1mv3_arcface_r50_fp16
$
$ # Get the pretrain model of arcface and put it on folder /app/arcface_torch/backbones/ms1mv3_arcface_r50_fp16
$ wget https://github.com/vietbacnguyen96/AI_Awards_Server/releases/download/v1.0.0/backbone.pth
$ mkdir .\app\arcface_torch\backbones\ms1mv3_arcface_r50_fp16
$ mv backbone.pth /app/arcface_torch/backbones/ms1mv3_arcface_r50_fp16
$
$ # Install python 3.6.2
$ https://www.python.org/downloads/release/python-362/
$
$ # Install torch 1.10.0
$ https://pypi.org/project/torch/1.10.0/#files
$
$ # Install virtualenv 
$ pip install virtualenv
$
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env -p python3.6
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ virtualenv env -p python3.6
$ .\env\Scripts\activate
$
$ # Install modules
$ pip3 install -r requirements.txt
$
$ # Install torch
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$
$ # Start the application (development mode)
$ python3 gui.py
$
```