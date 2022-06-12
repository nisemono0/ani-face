# Anime face detector
<p align="center">Example</p>
<p align="center"> <img src="https://user-images.githubusercontent.com/12276968/172742349-707d8aa6-1d36-4780-b607-8a45910b1110.png" width="320"/> </p>

Trained on [this dataset](https://www.kaggle.com/datasets/andy8744/annotated-anime-faces-dataset)<br>
Following [this paper](https://pjreddie.com/media/files/papers/yolo_1.pdf)<br>


# How to use
Run ```pip install -r requirements.txt``` <br>
Download the model from [here](https://github.com/nisemono0/ani-face/releases) and move it to the ```models``` directory <br>
Run ```python run.py PATH_TO_IMAGE``` to run on an image or ```python -h, --help``` for a list of arguments

```
 run.py [-h] [-m MODEL] [-i IOU_THRESHOLD] [-d DROP_THRESHOLD]
              [-b BORDER] [-s SAVE]
              IMAGE
```


# To train
Make sure the ```data``` folder has this structure
```
.
├─ ...
├─ data
│  │  test.csv
│  │  train.csv
│  ├─ test
│  │  ├─images
│  │  │       1.jpg
│  │  │       2.jpg
│  │  │      
│  │  └─labels
│  │          1.txt
│  │          2.txt    
│  └─ train
│     ├─images
│     │       a.jpg
│     │       b.jpg
│     │      
│     └─labels
│             a.txt
│             b.txt
└─ ...
```
```train.csv``` and ```test.csv``` have the following format
```
image,label
1.jpg,1.txt
2.jpg,2.txt
...
```

Run ```python train.py``` to train with the default arguments or ```python train.py -h, --help``` to see help <br>
```
train.py [-h] [-s SPLIT_SIZE] [-b NUM_BOXES] [-c NUM_CLASSES]
```
Run ```python test.py``` to test <br>

# Note
You can edit the ```config/config.py``` file to change hyperparameters and tweak other settings