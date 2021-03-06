# Offline Handwritten Mathematical Symbols Reorganization using CNN
Using Image Processing Project contains different jupyter notebooks and evaluation results.These notebooks can be directly opened in colab on click of button. Also it contains the information about the connecting to kaggle to get the dataset directly into colab working directory.

Report:
Group members: 

1)Hinakoushar Tatakoti

2)Hafiz M Amir Hussain



[**1)Introduction and Goal:**](#-introduction-and-goal)

[**2)Motivation:**](#-motivation)

[**3)Abstract:**](#abstract)

[**4)Connected to kaggle**](#connected-to-kaggle)

[**5)Dataset:**](#dataset)

[**6)Sample Size:**](#sample-Size)

[**7)Tools used:**](#tools-used)

[**8)Task Diagram:**](#task-Diagram)

[**9)Model 2- Pretrained mobileNetV2:**](#model-2--pretrained-mobilenetv2)

[**10)Model-3 -Trained from Scartch:**](#model-3--trained-from-scartch)

[**11)Summary:**](#summary)

[**12)References:**](#references)


# Introduction and Goal

With the advancement of technology., computers become more human like. It naturally makes sense for them to be able to read human writing. This project was developed to experiment with a computer’s ability to read human handwriting i.e. inconsistent styles and patterns with same meaning. This project aligns with computer vision, pattern recognition and artificial intelligence and employs machine learning and image processing techniques.However, offline handwritten recognition is used for the scanned documents; it is less appealing than online.
Competition on Recognition of Online Handwritten Mathematical Expressions(CROHME) 2016 concluded that the recognition of HMEs was still a challenge after six years of competition.The online method works well for connected strokes or cursive strokes, while the offline way can overcome the problem
of out-of-order strokes or duplicated strokes using contextual information.
Goal is to Increase the accuracy,Insert the more layers to existing model.Use of pretrained model mobileNetV2 for 45*45 image size with increased filter or same filter size as existing.Different Hyperparameters to tune the model [learning rate(0.01- 0.00001), optimization function (ADAM and SGD), activation functions (RelU, tanH), loss functions (softmax and sigmoid)]

# Motivation

Due to the technological advances in recent years, paper scientific documents are used less and less. Thus, the trend in the scientific community to use digital documents has increased considerably. Among these documents, there are scientific documents and more specifically mathematics documents.
So this project give a basic prediction of mathematical symbol, to research recognizing handwritten math language in variety of applications.The main motivation for this work is both recognizing of the handwritten mathematical symbol, digits and characters which can be further be used for mathematical expression recognition.

# Dataset

Source: https://www.kaggle.com/xainano/handwrittenmathsymbols
It includes basic Greek alphabet symbols like: 
alpha, beta, gamma, mu, sigma, phi and theta.English alphanumeric symbols are included.All math operators, set operators.Basic pre-defined math functions like: log, lim, cos, sin, tan.Math symbols like: \int, \sum, \sqrt, \delta.

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/dataset.jpg)

# Abstract

Offline Handwritten math symbols recognization using project,inspired by dataset on kaggle(https://www.kaggle.com/xainano/handwrittenmathsymbols).Thesesysmbols are parsed, extracted and modified from inkML of CROHME(http://www.isical.ac.in/~crohme/index.html) dataset.
There are 82 classes of mathematical symbols including characters and digits that makes 375974 images. 
The handwritten symbol as an input image and output should predict and classify the symbol from 82 classes available in the dataset.

# Connected to kaggle

Directly dataset is downloaded from kaggle and extracted into google colab using below steps


![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/connecting_kaggle.jpg)


# Sample Size

The available data set for training and prediction are 

special chars ---> '!', '(', ')', '+', ',', '-' ,'=','[', ']','{','}'

numbers --->  '0', '1' ,'2', '3', '4', '5', '6' ,'7', '8', '9' 

Capital Letters ---> 'A','C','G', 'H' ,'M' ,'N' ,'R', 'S', 'T', 'X'

Small Letters ---> 'b' ,'d','e','f','j' ,'k', 'l', 'o' ,'p','q','u','v','w','y','z' 

Maths symbol---> 'alpha','ascii_124','beta' ,'cos','exists','forall','forward_slash','gamma', 'geq', 'gt' ,'in', 'infty', 'int','lambda' ,'ldots',
 'leq', 'lim', 'log', 'lt', 'mu', 'neq','lambda' ,'ldots','leq', 'lim' ,'log', 'lt', 'mu' ,'neq','rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times'

# Tools Used

Google colab + kaggle + github

# Task Diagram

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/Recognization_Model.jpg)

# Model 1- Training from scratch

Model is build from scartch using 1 input layer, 2 hidden layers and one output layer.
used kernel size 2*2

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-sketch.jpg)

**Model Layers**

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-1.jpg)

**Hyperparameter Optimization**

EPOCHS = 50
BS = 100 #Batch size
LR = 1e-3 #Learning rate 0.001
img_dim = (45,45,3)
Time taken : 130 mins total

Accuracy : 

loss: 0.3292 - accuracy: 0.8944 - val_loss: 0.1657 - val_accuracy: 0.9512

Model Evaluation graphs:

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-1_model_accuracy.jpg)

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-1-Model-loss.jpg)

**Prediction**

Able to recognize 20 symobols out of 82 symobols
labels which are recognized are:
lab = ['tan'  '='  'l' 'log'  '-' 'pm'  'X'
  'S' 'div' '0' '1' 'lambda' 'k' 'phi' 'd'  'f' 'geq' 'infty' 'leq' 'lim' ')'
  'T'  'y' '3'  '7' 'gt' 'pi'  'sum' 'ldots' 'M' 'j' 'sin' 'lt' 'e' 'G']
 
 
 ![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-1-predictions.jpg)


# Model 2- Pretrained mobileNetV2

Model is pretrained on MobileNetV2 with imagenet weights 

base_model=tf.keras.applications.MobileNetV2( include_top=False, input_shape=(128,128,3), pooling='max', weights='imagenet')

**Model Layers**

 ![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-2-summary.jpg)

**Hyperparameter Optimization**
the last layer modified added, BatchNormalization, Dense layer, Dropout layer and final fully connected layer
Epoch: 5
Time:60 mins for each epoch
learning_rate=.001

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-2-training-accuracy.jpg)


**model evaluation**

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-2-model-accuracy.jpg)

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-2-model-loss.jpg)


**Prediction**
Prediction for the pretrained on MobileNetV2 with imagenet weights quite not quite.Eventhough accuracy is very good achieved 97% but prediction probability alomost between 5-10 for every images, only abale to recognize few symobols.

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-2-predictions.jpg)



# Model-3 -Trained from Scartch

As pretrained model is not predicting quite well, i had to choose training from scartch using CNN, i have used 1 input layer , 2 sub sampling layer and 2 fully connected layers.Total 210,494 trainiable parameters none non-trainable parameters.


**Model Layers**

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-3-summary.jpg)


**Hyperparameter Optimization**
total 3 epoch is used
Image dim: (45,45,1)
learning_rate = 5e-4 = 0.0005
Time taken: 30 mins for each epoch

**Model evaluation**

training accuracy: 
loss: 0.3323 - accuracy: 0.8979 - val_loss: 0.3079 - val_accuracy: 0.9053
![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-3-training-accuracy.jpg)


test Evaluation:
loss: 0.3079 - accuracy: 0.90580.3079117238521576 0.9057754278182983

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-3-evaluation.jpg)


**Prediction**

Prediction quite better than model-2, as model-3 is able to predict 15-20 symbols accurately with high probability rate.

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/model-3-predictions3.jpg)

# Summary

* Trained from scratch model recognized better than pretrained model.
* Pretrained model using MobileNetV2 ( weights imagnet) not quite good prediction.
* The higher dimension images not performed well during predictions.
* Model with 3 Conv layers worked better.
* More GPU power to try with different hyperparameters.
* Google colab often disconnects the session it’s problematic for training.
* Drawback faced, saving a model and weights directly into Github not possible.


# References

https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

https://towardsdatascience.com/creating-a-trashy-model-from-scratch-with-tensorflow-2-an-example-with-an-anomaly-detection-27f0d1d7bd00

https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

https://www.kaggle.com/xainano/handwrittenmathsymbols

https://github.com/ThomasLech/CROHME_extractor

https://stackoverflow.com/questions/67266161/how-to-train-amd-test-dataset-of-images-downloaded-from-kaggle

https://github.com/RichmondAlake/tensorflow_2_tutorials/blob/master/13_lenet-5.ipynb






