# Offline Handwritten Mathematical Symbols Reorganization using CNN
Using Image Processing

Report:
Group members: 

1)Hinakoushar Tatakoti
2)Hafiz M Amir Hussain

[**1)Introduction and Goal:**](#introduction-and-goal)

[**2)Motivation:**](#Motivation)

[**3)Abstract:**](#Abstract)

[**4)Dataset:**](#Dataset)

[**5)Sample Size:**](#Sample-Size)

[**6)Tools used:**](#Tools-used)

[**7)Task Diagram:**](#Task-Diagram)

**Introduction and Goal**(#introduction-and-goal)

With the advancement of technology., computers become more human like. It naturally makes sense for them to be able to read human writing. This project was developed to experiment with a computer’s ability to read human handwriting i.e. inconsistent styles and patterns with same meaning. This project aligns with computer vision, pattern recognition and artificial intelligence and employs machine learning and image processing techniques.However, offline handwritten recognition is used for the scanned documents; it is less appealing than online.
Competition on Recognition of Online Handwritten Mathematical Expressions(CROHME) 2016 concluded that the recognition of HMEs was still a challenge after six years of competition.The online method works well for connected strokes or cursive strokes, while the offline way can overcome the problem
of out-of-order strokes or duplicated strokes using contextual information.
Goal is to Increase the accuracy,Insert the more layers to existing model.Use of pretrained model mobileNetV2 for 45*45 image size with increased filter or same filter size as existing.Different Hyperparameters to tune the model [learning rate(0.01- 0.00001), optimization function (ADAM and SGD), activation functions (RelU, tanH), loss functions (softmax and sigmoid)]

**Motivation**

Due to the technological advances in recent years, paper scientific documents are used less and less. Thus, the trend in the scientific community to use digital documents has increased considerably. Among these documents, there are scientific documents and more specifically mathematics documents.So this project give a basic prediction of mathematical symbol, to research recognizing handwritten math language in variety of applications.The main motivation for this work is both recognizing of the handwritten mathematical symbol, digits and characters which can be further be used for mathematical expression recognition.

**Dataset**

Source: https://www.kaggle.com/xainano/handwrittenmathsymbols
It includes basic Greek alphabet symbols like: 
alpha, beta, gamma, mu, sigma, phi and theta.English alphanumeric symbols are included.All math operators, set operators.Basic pre-defined math functions like: log, lim, cos, sin, tan.Math symbols like: \int, \sum, \sqrt, \delta.

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/dataset.jpg)

**Abstract**

Offline Handwritten math symbols recognization using project,inspired by dataset on kaggle(https://www.kaggle.com/xainano/handwrittenmathsymbols).Thesesysmbols are parsed, extracted and modified from inkML of CROHME(http://www.isical.ac.in/~crohme/index.html) dataset.
There are 82 classes of mathematical symbols including characters and digits that makes 375974 images. 
The handwritten symbol as an input image and output should predict and classify the symbol from 82 classes available in the dataset.

**Sample Size**

The available data set for training and prediction are 

special chars ---> '!', '(', ')', '+', ',', '-' ,'=','[', ']','{','}'

numbers --->  '0', '1' ,'2', '3', '4', '5', '6' ,'7', '8', '9' 

Capital Letters ---> 'A','C','G', 'H' ,'M' ,'N' ,'R', 'S', 'T', 'X'

Small Letters ---> 'b' ,'d','e','f','j' ,'k', 'l', 'o' ,'p','q','u','v','w','y','z' 

Maths symbol---> 'alpha','ascii_124','beta' ,'cos','exists','forall','forward_slash','gamma', 'geq', 'gt' ,'in', 'infty', 'int','lambda' ,'ldots',
 'leq', 'lim', 'log', 'lt', 'mu', 'neq','lambda' ,'ldots','leq', 'lim' ,'log', 'lt', 'mu' ,'neq','rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times'

**Tools Used** Google colab + kaggle + github

**Task Diagram**

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/Recognization_Model.jpg)

**Model 1- Training from scratch**

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





