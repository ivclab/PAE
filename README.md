# Packing and Expanding (PAE)
official implementation of [Increasingly Packing Multiple Facial-Informatics Modules in A Unified Deep-Learning Model via Lifelong Learning](https://dl.acm.org/citation.cfm?id=3325053)

Created by [Steven C. Y. Hung](https://github.com/fevemania), [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), [Timmy S. T. Wan](https://github.com/bigchou), [Chein-Hung Chen](https://github.com/Chien-Hung), [Yi-Ming Chan](https://github.com/yimingchan), Chu-Song Chen

## Introduction
Simultaneously running multiple modules is a key requirement for a smart multimedia system for facial applications including face recognition, facial expression understanding, and gender identification. To effectively integrate them, a continual learning approach to learn new tasks without forgetting is introduced. Unlike previous methods growing monotonically in size, our approach maintains the compactness in continual learning. The proposed packing-and-expanding method is effective and easy to implement, which can iteratively shrink and enlarge the model to integrate new functions. Our integrated multitask model can achieve similar accuracy with only 39.9% of the original size.

## Citing Paper
Please cite following paper if these codes help your research:

    @inproceedings{hung2019increasingly,
        title={Increasingly Packing Multiple Facial-Informatics Modules in A Unified Deep-Learning Model via Lifelong Learning},
        author={Hung, Steven CY and Lee, Jia-Hong and Wan, Timmy ST and Chen, Chein-Hung and Chan, Yi-Ming and Chen, Chu-Song},
        booktitle={Proceedings of the 2019 on International Conference on Multimedia Retrieval},
        pages={339--343},
        year={2019},
        organization={ACM}
    }

## Prerequisition
- Ubuntu
- Python 3
- Cuda-9.0 ([Google drive](https://drive.google.com/file/d/1eu3Pstdyhs3cg-brHrsPMgx_LMFoplSs/view?usp=sharing))

(1) If the operation system of your computer is Ubuntu 18.04, you need to follow the command to downgrade your complier:
```bash
$ sudo apt install gcc-6 g++-6
$ sudo ln -s /usr/bin/gcc-6 /usr/local/bin/gcc
$ sudo ln -s /usr/bin/g++-6 /usr/local/bin/g++
```

(2) Set the environmental variable of Cuda 9.0:
```bash
$ vi ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/extras/CUPTI/lib64
$ source ~/.bashrc
```
- Cudnn v7 ([Google drive](https://drive.google.com/file/d/1sTxprxbW1GoLHXNjJVh0Jqefb_IPDu9X/view?usp=sharing))
```bash
$ tar -xzvf cudnn-9.0-linux-x64-v7.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
$ cd /usr/local/cuda-9.0/lib64
$ sudo ln -sf libcudnn.so.7.0.5 libcudnn.so.7
$ sudo ln -sf libcudnn.so.7 libcudnn.so
$ sudo ldconfig
```
- [TensorFlow](https://www.tensorflow.org/install/install_linux)
```bash
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ sudo pip3 install -U virtualenv
$ virtualenv --system-site-packages -p python3 ./tfvenv
$ source ./tfvenv/bin/activate
$ pip install tensorflow-gpu==1.7
```
- other python's library
```bash
$ pip install -r requirement.txt
```

## Usage
Clone the PAE repository:
```bash
$ git clone --recursive https://github.com/ivclab/PAE.git
```

### Experiment One (Face Verification, Gender and Age Modules)
1. Download Vggface2(image size is 182x182), LFW(image size is 160x160) and Adience datasets(image size is 182x182) which have been aligned by [MTCNN](https://github.com/ivclab/PAE/tree/master/src/align)
```bash
$ cd data
$ python download_aligned_LFW.py
```

2. Download all epoches of PAENet models and baseline models in all experiment (The file size of official_checkpoint.zip is 78 GB)
```bash
$ python download_official_checkpoint.py
```

3. Inference the PAENet model
- inference the face task of PAENet model and its accuracy and model size stored in [accresult/experiment1/PAENet_face](https://github.com/ivclab/PAE/blob/master/accresult/experiment1/PAENet_face.csv). The accuracy and model size of baseline FaceNet stored in [accresult/baseline/experiment1/FaceNet](https://github.com/ivclab/PAE/blob/master/accresult/baseline/experiment1/FaceNet.csv)
```bash
$ bash src/inference_first_task.sh
```
- inference the age and gender tasks of PAENet model, their accuracy and model size stored in [accresult/experiment1/age]() and [accresult/experiment1/gender](). The accuracy and model size of baseline AgeNet stored in [accresult/baseline/experiment1/AgeNet](). 

4. The accuracy and model size of our PAENet in experiment one in [accresult/experiment1](https://github.com/ivclab/PAE/tree/master/accresult/experiment1) and [accresult/facenet](https://github.com/ivclab/PAE/tree/master/accresult/facenet)

### Experiment Two (Face Verification, Gender and Expression)

4. The accuracy and model size of our PAENet in experiment two in [accresult/experiment2](https://github.com/ivclab/PAE/tree/master/accresult/experiment2) and [accresult/facenet](https://github.com/ivclab/PAE/tree/master/accresult/facenet)
    
#  Coming Soon ...

## [Compacting, Picking and Growing (CPG)](https://github.com/ivclab/CPG)
We enhance our PAE to become the CPG, which is published in NeurIPS, 2019.

## Reference Resource
- [FaceNet](https://github.com/davidsandberg/facenet)

## Contact
Please feel free to leave suggestions or comments to [Steven C. Y. Hung](https://github.com/fevemania), [Jia-Hong Lee](https://github.com/Jia-HongHenryLee)(honghenry.lee@gmail.com), [Timmy S. T. Wan](https://github.com/bigchou), [Chein-Hung Chen](https://github.com/Chien-Hung), [Yi-Ming Chan](https://github.com/yimingchan), Chu-Song Chen(song@iis.sinica.edu.tw)

