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

## Implementaçao

https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8

isso pode ser util ou ent nao mas fica aqui na mma