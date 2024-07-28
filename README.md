 # This repository is the official implementation of  "PSSTRNET: PROGRESSIVE SEGMENTATION-GUIDED SCENE TEXT REMOVAL NETWORK" 
The pretrained models on SCUT-Syn and SCUT-EnsText.  [[BaiduDisk](https://pan.baidu.com/s/1WhMUeI2CPNGsKJSCghuKrA?pwd=m5o6)] [[GoogleDrive](https://drive.google.com/file/d/1YpxKbFQnzO2jcDd2nzEwsJWgmDq-S0Ty/view?usp=share_link)]

The visualization results on SCUT-Syn and SCUT-EnsText.  [[BaiduDisk](https://pan.baidu.com/s/1AFfFodf1DufiflbEA8caMw?pwd=nph4)] [[GoogleDrive](https://drive.google.com/file/d/1hwt_gNzii0KkrqFP6GhpDHd3am4s7eOu/view?usp=share_link)]

The evaluation metrics can refer the EraseNet. [EraseNet](https://github.com/lcy0604/EraseNet)


# Inference

0. Create a virtual env:
```bash
conda create -n psstr pyhton=3.10 -y
```
```bash
conda activate psstr
```

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the command:
```bash
python inference.py -i /path/to/your/images
```

**NOTE:** you can run with one or several images.