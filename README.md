# Identify text/non text #

Run the text_non_text.py script passing the name of the directory, which contains the dataset , as a parameter. The script will use a convolutional neural network to classify the images from the dataset between with text e nontext. After the classification, the script will copy the images into 2 subdiretories, `DATASET_PROCESSADO/COM_TEXTO` and `DATASET_PROCESSADO/SEM_TEXTO`, according to the respective output of the CNN.

## Example ##

```
text_non_text.py
dataset
└── images
    └── Img1.jpg
    └── Img2.jpg
    └── Img3.jpg
    └── ...
```

Following the example of the strucuture of the diretories given above, the execution of the script will be:

``` 
python3 text_non_text.py dataset
```

Don't forget to store the images inside a subdiretory. The parameter passed to the script must be `dataset`, following the example above, not `dataset/images`.

## Requirements ##

 - Keras 2.5.0
 - Numpy 1.19.5
 - Tensorflow 2.5.0
 - Sklearn 0.24.2
 - Matplotlib 3.4.2
 - gdow 3.13.0
 - tf-nightly 2.6.0


## Results ##
The CNN used is a [MobileNet](https://keras.io/api/applications/mobilenet/), basead on publication [[1]](#1). An accuracy of 92,4% was obtained during the validation step and using a dataset of 1000 images.

## Reference
<a id="1">[1]</a> 
Bai, Xiang, et al. "Text/non-text image classification in the wild with convolutional neural networks." Pattern Recognition 66 (2017): 437-446
