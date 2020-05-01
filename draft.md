# Face Recognition

* Tiago Melo
* João Nogueira

## Abstract

The goal of this paper is to study and compare different Machine Learning models and their performances in the context of the classical problem of face recognition. The benchmarked models include Linear Regression, Logistic Regression, Support Vector Machines (SVM) and Neural Networks. After implementing and testing each of these, one is chosen and its hyper-parameters are finely tuned. We aim to present a system that is able to identify and recognize a subject's head in near-real-time without too much added complexity. In order to do this <BRIEFLY_DESCRIBE_METHOD>

## Introduction

Face recognition has proven to be of more and more use as technology advances. Real-time identification of people, further security when unlocking personal devices or even implementing a social credit point system like in China :) Unlike face detection, which focuses on understanding where a face is on a given picture, face recognition focuses on understanding who the face in a picture belongs to; in other words, identifying people.

The model that we use here is based on Hidden Markov Models and to extract features from images we use Singular Value Decomposition (SVD). You
http://www.facerecognitioncode.com can think about the model as a black box that has two parts. The first part is for training. We give the model images of a number of people. Let’s say we give it 5 images for each person. This part is called training and the model learns to recognize these people. Then, the second part of the model is for testing. In testing we give the model, one image that we don’t know who he/she is. The model returns a probability for each person that it was trained on. For example if the model has learned 40 people then it gives us 40 probabilities. Each probability indicates how likely the input image could be that person. Then we simply can say that the person that returns the highest probability is in fact the person that is on the input image.

## Section I, State of the Art Review

Here we  will be analyzing five papers on the subject of face recognition,

ORL database with SVM:
https://www.researchgate.net/publication/2427763_Face_Recognition_by_Support_Vector_Machines

ORL database with CNN:
https://www.cs.cmu.edu/afs/cs/user/bhiksha/WWW/courses/deeplearning/Fall.2016/pdfs/Lawrence_et_al.pdf

https://www.researchgate.net/publication/332865261_Machine_Learning_approaches_for_Face_Identification_Feed_Forward_Algorithms

https://www.researchgate.net/publication/300795728_Deep_learning_and_face_recognition_the_state_of_the_art

M. Turk and A. Pentland :
https://www.cin.ufpe.br/~rps/Artigos/Face%20Recognition%20Using%20Eigenfaces.pdf

Dataset:
 F. Samaria and A. Harter
  "Parameterisation of a stochastic model for human face identification"
  2nd IEEE Workshop on Applications of Computer Vision
  December 1994, Sarasota (Florida).

## Data

The data were loaded into a 410 * (112 * 92 + 1) matrix. Essentially, we are loading a flattened image on each row along the folder it belongs to. The folder number will act as a label.

### Data Splitting

Since when reading data from folders it is loaded orderly, i. e., most labels were still grouped up, the data were shuffled and only then split in train, test and validation sets (60 20 20). This was to ensure every label appeared at least once while training our model. @Joao-Nogueira-gh escreve mais coisas aqui nsei oq


## Implementaçao

https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8

isso pode ser util ou ent nao mas fica aqui na mma


## Picking a Model

Like explained above, we set out to try a set of machine learning models, pick the best-performing one and fine tune it. The models were either implemented by hand, using libraries or both, like Logistic Regression. 
 
### Linear Regression

One of the simplest Machine Learning models, Linear Regression consists of tweaking a set of parameters (Thetas) until they converge into a set of values that best translates the real world. This is called training. Since in our example we have 10304 pixels, our model will consist of a one by 10304 values, each corresponding to the correct weight each pixel should have in order to accurately predict classes. Since this model is usually used in continuous contexts and outputs a real number, we validated it by casting the output to an integer and comparing it with the right folder number, also cast to an integer.

In order to do accomplish this, we focus on minimizing a function that allows us to understand how "far" our model is from reality. By finding the set of thetas that allows this minimum cost function, we achieve the parameters that bring our model the closest to reality. In our case, the cost function consists of multiplying our set of parameters by the input, resulting in matrix called Hypothesis. We then calculate the element-wise difference between our Hypothesis and the real values it should assume, y, square it, to ensure positive values and calculate the mean. This is called the Mean Squared Error method and is one of the most popular choices for cost calculation.

There is a number of algorithms that aim to reduce this cost function in the best way possible. These are called optimizers. The optimizer chosen for this model was the Gradient Descent. In Calculus, the gradient can be interpreted as the direction or rate of the fastest increase. Hence, it's inverse can be used to find a minimum of any function, namely the Cost Function. Naturally, this choice has a few caveats, such as local minima.

Linear Regression was implemented by hand. Based on course code, we can pick a number of iterations (i. e. how many times the gradient will "descend") and a learning rate, how much it does descend every iteration. As explained, on every iteration we measure the cost function of the current thetas, calculate the gradient for the current position and tweak them in the direction told by the gradient, iteratively updating our parameters and, hopefully, dragging them closer to an accurate, real world prediction model. However, it is worth mentioning this model performs best for continuous values and not classification problems.

As expected, this model performed pretty poorly predicting classes, obtaining only 1.2% accuracy on the validation set.

### Logistic Regression

For a classification problem, a much more suitable choice for a model would be Logistic Regression. It differs only on a few key concepts when compared to Linear Regression, such as the cost function calculation. For instance, whereas in Linear Regression the cost is calculated as explained above, in logistic regression the cost function is calculated by increasing the value of the cost the closer it is to the opposite label. This assumes a binary model, however. Since the task at hand consists of a multiclass classification problem, we must adopt a fitting strategy, such as One Versus All. This consists of training one binary classifier for each existing class. Later, for predictions, the winner takes all: the binary classifier with the highest output score assigns the class.

The rest is similar. We have a set of parameters we aim to reduce. When making predictions we multiply the resulting theta vector by our parameters and it outputs a list of numbers that can be translated to probabilites using a function such as softmax, where each represents the probability that this input corresponds to each idk label. 

Much like Linear Regression, Logistic Regression was also implemented by hand, based on course code. There is a set of tweakable hyper-parameters, such as number of iterations and learning rate, that can be finely tuned. The one-vs-all strategy, however, was implemented by training one binary classifier per class. 

### Support Vector Machines

Consist of setting boundaries between classes. <TODO_READ_SLIDES_PASTE_HERE>
