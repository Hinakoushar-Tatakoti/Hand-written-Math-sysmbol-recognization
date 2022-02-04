# Offline Handwritten Mathematical Symbols Reorganization using CNN
Using Image Processing

Report:
Group members: 
1)Hinakoushar Tatakoti
2)Hafiz M Amir Hussain

Motivation:
Due to the technological advances in recent years, paper scientific documents are used less and less. Thus, the trend in the scientific community to use digital documents has increased considerably. Among these documents, there are scientific documents and more specifically mathematics documents.So this project give a basic prediction of mathematical symbol, to research recognizing handwritten math language in variety of applications.The main motivation for this work is both recognizing of the handwritten mathematical symbol, digits and characters which can be further be used for mathematical expression recognition.

Dataset:
Source: https://www.kaggle.com/xainano/handwrittenmathsymbols
It includes basic Greek alphabet symbols like: 
alpha, beta, gamma, mu, sigma, phi and theta.English alphanumeric symbols are included.All math operators, set operators.Basic pre-defined math functions like: log, lim, cos, sin, tan.Math symbols like: \int, \sum, \sqrt, \delta.

![alt text](https://github.com/Hinakoushar-Tatakoti/Hand-written-Math-sysmbol-recognization/blob/master/images/dataset.jpg)

Abstract:
Offline Handwritten math symbols recognization using project,inspired by dataset on kaggle(https://www.kaggle.com/xainano/handwrittenmathsymbols).Thesesysmbols are parsed, extracted and modified from inkML of CROHME(http://www.isical.ac.in/~crohme/index.html) dataset.
There are 82 classes of mathematical symbols including characters and digits that makes 375974 images. 
The handwritten symbol as an input image and output should predict and classify the symbol from 82 classes available in the dataset.

The available data set for training and prediction are 

special chars ---> '!' '(' ')' '+' ',' '-' ,'=','[' ']','{','}'
-------------------------
numbers --->  '0' '1' '2' '3' '4' '5' '6' '7' '8' '9' 
-------------------------
Capital Letters ---> 'A','C','G' 'H' 'M' 'N' 'R' 'S' 'T' 'X'
-------------------------
Small Letters ---> 'b' ,'d','e','f','j' 'k' 'l', 'o' ,'p','q','u', 'v', 'w','y','z' 
Maths symbol---> 'alpha', 'ascii_124','beta' ,'cos','exists','forall','forward_slash','gamma', 'geq', 'gt' ,'in', 'infty', 'int','lambda' ,'ldots',
 'leq', 'lim', 'log', 'lt', 'mu', 'neq','lambda' ,'ldots','leq', 'lim' ,'log', 'lt', 'mu' ,'neq','rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times'
 -------------------------
 
