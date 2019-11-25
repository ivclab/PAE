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
- Python 3
- [TensorFlow](https://www.tensorflow.org/install/install_linux)
```bash
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ sudo pip3 install -U virtualenv
$ virtualenv --system-site-packages -p python3 ./tfvenv
$ source ./tfvenv/bin/activate
$ pip install tensorflow-gpu==1.15
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
### Experiment One
1. Download Vggface2, LFW and adience datasets which have been aligned by [MTCNN](https://github.com/ivclab/PAE/tree/master/src/align)
```bash
$ 
```
    
#  Coming Soon ...

## [Compacting, Picking and Growing (CPG)](https://github.com/ivclab/CPG)
We enhance our PAE to become the CPG, which is published in NeurIPS, 2019.

## Contact
Please feel free to leave suggestions or comments to [Steven C. Y. Hung](https://github.com/fevemania), [Jia-Hong Lee](https://github.com/Jia-HongHenryLee)(honghenry.lee@gmail.com), [Timmy S. T. Wan](https://github.com/bigchou), [Chein-Hung Chen](https://github.com/Chien-Hung), [Yi-Ming Chan](https://github.com/yimingchan), Chu-Song Chen(song@iis.sinica.edu.tw)

