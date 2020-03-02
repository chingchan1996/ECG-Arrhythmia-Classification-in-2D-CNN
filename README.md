# ECG-Arrhythmia-Classification-in-2D-CNN
This is an implementation based on this paper, **"ECG arrhythmia classification using a 2-D convolutional neural network", Tae Joon Jun et al., CVPR 2018."** with some personal modifications

# **Dataset**
This repo adapts [MIT-BIH Arrhythmia Database](https://physionet.org/physiobank/database/mitdb/) as training and testing dataset.

## **Download Dataset**
The download file can be found under ````/mit_arrythmia_dat/````
>dataset_downloader.sh

the script is modified from [MIT-BIH-Arrhythmia-Downloader](https://github.com/lext/MIT-BIH-Arrhythmia-Downloader.git)

## **Data Pre-Process**
To turn ECG signals to images, the script can be found under ```/mit_arrhythmua_dat/```
>readDataset.py

The script will output EIGHT types of heart beats based on the annotations that the officials provide. A heart beat is defined by the peak of R waves
according to the contents on the websites.

#### **Difference from the paper - The adapt formula to produce images**

1. According to the paper, each images is form by the following formula

    ```T(Qpeak(n − 1) + 20) ≤ T(n) ≤ T(Qpeak(n + 1) − 20)``` 

    where n is the number of peak and 20 is the time slice.

    Following this formula to produce images results in that the length of each image is not constent. Therefore,
    in this repo the formula is changed to the following,

    ```T(Qpeak(n) - 96) ≤ T(n) ≤ T(Qpeak(n) + 96)```

2. The output size of images are **196x128** instead of **128x128** proposed in the paper.


#### **Difference from the paper - Data preprocess**

According to the paper, each image is cropped and resized to 9 versions.
...

There are more contents to be fulfilled in the future.

### **THIS REPO IS LIKELY NOT UPDATED IN THE FUTURE DUE TO LOST OF RESEARCHED DATA**
The main conclusion for this work is the insuffiencient opensource dataset leads to the overfitting to the trained model. When considering this topic, the researchers should be awared of the limited training resource.
