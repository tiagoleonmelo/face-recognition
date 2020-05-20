# Face Recognition

* Tiago Melo
* João Nogueira

## Abstract

The goal of this paper is to study and compare different Machine Learning models and their performances in the context of the classical problem of face recognition. The benchmarked models include Linear Regression, Logistic Regression, Support Vector Machines (SVM), Neural Networks and Decision Trees. After implementing and testing each of these, one is chosen and its hyper-parameters are finely tuned. We aim to present a system that is able to identify and recognize a subject's head in near-real-time without too much added complexity. In order to do this, every trained model will be shortly explained, as well as the corresponding development strategy. Furthermore, the data are gonna be studied and analyzed beforehand in order to more easily explain and troubleshoot problems when discussing results. This pre-processing consists of, essentially, plan ahead what classes are similar and more likely to be mislabelled and, in turn, lower the model's accuracy, recall and precision. Lastly, we will make use of Eigenfaces to reduce example dimensions and have a more efficient and faster system that is able to recognize faces via Principal Component Analysis. By doing this in the Exploratory Data Analysis, we will also have a better grasp of the data and the images that are most prone to being wrongly labelled by being too similar.

## Introduction

Face recognition has proven to be of more and more use as technology advances. Real-time identification of individuals, further security when unlocking personal devices or implementing a social credit point system. Unlike face detection, which consists of identifying where in a picture can faces be spotted [Omid no PPT q vinha com o DataSet], face recognition focuses on understanding who the face in a picture belongs to; in other words, identifying people. As mentioned before, this project aims to propose and develop a well-performing Machine Learning model that is able to recognize and identify faces with acceptable efficiency. This paper has the goal of explaining the thought process behind the development of said project, the obtained results when testing, validating and comparing different models and how feature extraction via PCA can prove to be of extreme relevance on such projects.

This article is structured in the following manner: we will start by briefly reviewing and analysing past work done in similar such problems using similar, or even the same, datasets in the State of the Art Review section. In the Data section, we will go through the content of the dataset, as well as explain its main features, how it was loaded, processed and split before it could be fed to our models. Additionally, we will also perform a short Exploratory Data Analysis by building the mean faces and Eigenfaces, which will allow us to find similar subjects and labels. Eigenfaces essentially consist of extracting the most relevant features regarding given subjects by means of Principal Component Analysis [paper das eigenfaces]. Once EDA has been completed, we will proceed towards the experimentation between different models for solving the task at hand in the Picking a Model section. Results will be briefly discussed and compared with other model's. In the Fine Tuning the Model section, we will pick a model and, as the name suggests, tweak the hyperparameters in order to maximize the model's performance. Lastly, we will discuss and display the results and conclusions obtained from this project, in the Results and Conclusions sections, respectively. 

## State of the Art Review

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

### Data description

The data used for the development of this project is the ORL Face Database. It consists of pictures of 40 people, where each person has had 10 photographs taken of them. The images have a height of 112 pixels by 92. They have been cropped and centered, so that the system can focus more on Facial Recognition, rather than an also popular problem in Machine Learning, Face Detection, as explained in the Introduction. Besides the people present in the original Dataset, a kind contributor has also added his pictures to the file system. Hence, our data consist of 410 112 by 92 greyscale images of 41 subjects, having 10 pictures each.

[SHOW HISTOGRAM]

That being said, the data were loaded into a 410 * (112 * 92 + 1) matrix. We want to have a matrix that holds a flattened version of the image on each row and has the name of the subject on its last column. Since our dataset, unfortunately, did not come with named subjects, we simply labelled each individual with the folder number the picture was stored in. An example of the resulting matrix can be seen in Figure 1.

[SHOW DATA.HEAD()]

The images were stored in the Portable GreyMap format. In other words, each pixel corresponded to an integer, ranging from 0 to 255, where the closer to 0 the darker the represented color is. After loading the data, it was normalized. By normalizing the information that we are working with, we are assuring we prune out any anomaly that we might find as well as ease the system performance and data load. The chosen method to normalize our data was to simply divide each pixel by 255, which resulted in each of these being converted into a number between 0 and 1. 

To make sure out data was properly loaded and parsed, we can randomly sample and display images in our dataset. Each of these has the corresponding label below them.

[SHOW IMAGES]

In order to load the data from the local folders into the script, we made use of Python's os library, which can traverse through the paths passed on as arguments recursively and retrieve the files in them. This resulted in most of the data being poorly mixed. In other words, same label images were bundled together in subsequent rows. Hence, when we split the data, the training set lacked a lot of information about other training examples. In order to prevent this, we shuffled the data beforehand, to dramatically increase the odds of such a scenario being near impossible. That being said, the split itself followed a regular distribution among sets: 60% (246 examples) for the training set, 20% (82 examples) for validation and testing sets.

### Exploratory Data Analysis

## Picking a Model

Like explained above, we set out to try a set of machine learning models, pick the best-performing one and fine tune it. The models were either implemented by hand, using libraries or both, like Logistic Regression. In order to more accurately pick a best-performing model, instead of simply training and validating one instance with randomly selected hyper-parameters, we tried to increase the odds of developing and picking a good model by creating and comparing three different sets of hyper-parameters. This way, we can pick the model that behaves the best based on its best set of hyper-parameters. This curation of hyper-parameters was only implemented on the custom methods, since the libraries dynamically adapt most of the initial hyper-parameters.

 
### Linear Regression

One of the simplest Machine Learning models, Linear Regression consists of tweaking a set of parameters (Thetas) until they converge into a set of values that best translates the real world. This is called training. Since in our example we have 10304 pixels, our model will consist of a one by 10304 values, each corresponding to the correct weight each pixel should have in order to accurately predict classes. Since this model is usually used in continuous contexts and outputs a real number, we validated it by casting the output to an integer and comparing it with the right folder number, also cast to an integer.

In order to do accomplish this, we focus on minimizing a function that allows us to understand how "far" our model is from reality. By finding the set of thetas that allows this minimum cost function, we achieve the parameters that bring our model the closest to reality. In our case, the cost function consists of multiplying our set of parameters by the input, resulting in matrix called Hypothesis. We then calculate the element-wise difference between our Hypothesis and the real values it should assume, y, square it, to ensure positive values and calculate the mean. This is called the Mean Squared Error method and is one of the most popular choices for cost calculation.

There is a number of algorithms that aim to reduce this cost function in the best way possible. These are called optimizers. The optimizer chosen for this model was the Gradient Descent. In Calculus, the gradient can be interpreted as the direction or rate of the fastest increase. Hence, it's inverse can be used to find a minimum of any function, namely the Cost Function. Naturally, this choice has a few caveats, such as local minima.

Linear Regression was implemented by hand. Based on course code, we can pick a number of iterations (i. e. how many times the gradient will "descend") and a learning rate, how much it does descend every iteration. As explained, on every iteration we measure the cost function of the current thetas, calculate the gradient for the current position and tweak them in the direction told by the gradient, iteratively updating our parameters and, hopefully, dragging them closer to an accurate, real world prediction model. However, it is worth mentioning this model performs best for continuous values and not classification problems.

We can see the variation of the each set of hyper-parameters' cost function in the plot in Fig. 1. We can naturally infer the best choice for our Linear Regression candidate would be model 4. However, as expected, since Linear Regression isn't fit for a classification problem, this model performed poorly predicting classes, obtaining only at best 2.1% accuracy on the validation set.

[INSERT LINREG PLOT]


### Logistic Regression

For a classification problem, a much more suitable choice for a model would be Logistic Regression. It differs only on a few key concepts when compared to Linear Regression, such as the cost function calculation. For instance, whereas in Linear Regression the cost is calculated as explained above, in logistic regression the cost function is calculated by increasing the value of the cost the closer it is to the opposite label. This assumes a binary model, however. Since the task at hand consists of a multiclass classification problem, we must adopt a fitting strategy, such as One Versus All. This consists of training one binary classifier for each existing class. Later, for predictions, the winner takes all: the binary classifier with the highest output score assigns the class.

The rest is similar. We have a set of parameters we aim to reduce. When making predictions we multiply the resulting theta vector by our parameters and it outputs a list of numbers that can be translated to probabilites using a function such as softmax, where each represents the probability that a given input corresponds to a given label. 

Much like Linear Regression, Logistic Regression was also implemented by hand, based on course code. There is a set of tweakable hyper-parameters, such as number of iterations and learning rate, that can be finely tuned. The one-vs-all strategy was also based on course code. We trained four Logistic Regression models with different sets of hyper-parameters. It is worth mentioning that, since we are making use of the One Vs All strategy, we will run all iterations per classifier. In other words, the total of iterations for a given set of hyperparameters will be equal to the number of classes (in our case 41) times the number of iterations. 

That being said, the best performing model got an accuracy of <TODO_FILL_HERE>. Each model's cost function can be seen in the plot below, with the corresponding set of hyper-parameters chosen. Although we can naturally infer all models converge to nearly the same value, we will pick the one that converges the fastest. The reason behind this choice is mainly since a fast-converging model means taking up less time for model training. [https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10]

[INSERT LOGREG PLOT]

On the other hand, the scikit-learn library contains an already built-in LogisticRegression model, we need only insert the initial parameters. For this case, we chose to only use the regularization parameter C, set to 1, which achieved a reasonably high validation accuracy: 95.16%. Currently, this is the best candidate for further fine-tuning and, lastly, testing.



### Support Vector Machines

An also fitting choice for a classification problem would be a Support Vector Machine (SVM). An SVM classifies data by finding the most adequate boundary that separates classes. This can be done by finding the boundary that maximizes the margin between the closest data point of each class to the border line. It is worth mentioning that only the closest points of each class, denominated by support vectors are used to determine the optimum margin between each label. In other words, for a two-class classiﬁcation problem, the goal is to separate the two classes by a function which is induced from available examples. Consider the examples in Fig. 1 (a),where there are many possible linear classiﬁers that can separate the data, but there is only one (shownin Fig. 1 (b)) that maximizes the margin(the distance between the hyperplaneand the nearest data point of each class). This linear classiﬁer is termed the optimal separating hyperplane (OSH). Intuitively, we would expect this boundary to generalize well as opposed to the other possible boundaries. [https://www.researchgate.net/publication/2427763_Face_Recognition_by_Support_Vector_Machines]

The cost function used in this model typically consists of an adaptation of the Logistic Regression cost function. Furthermore, it adds a regularization parameter C, which adjusts the penalty for misclassified training examples. Much like past models, the optimization objective of a SVM consists of finding a set of Thetas that brings the hypothesis as close as possible to reality, i. e., minimizes the cost function.

The library scikit-learn also offers a built-in SVM model, which needs only the initial hyper-parameters, such as C. For our model, we chose the default value for the penalization attribute: one. This resulted in a final validation accuracy of 98.78%. The tweakable hyper-parameters this model has will be further explained later on.


### Neural Networks

Similarly to other ML models, given a set of initial Thetas, our Neural Network will try to find the combination of Thetas that achieves the lowest cost function possible with a few caveats. A cost function, in its core, calculates how "far" the combination of Thetas is from the real world. By using the Vector Theta to calculate, in our example, the values of each pixel and calculating the Mean Square Error we will then have a useful metric that we wish to reduce to as low as possible, in order to have "edges" that allow our Network to learn and is able to accurately represent the real world.

There are a number of algorithms that aim to reduce the forementioned cost function, called optimizers. In this paper, we will be using the Gradient Descent. A gradient, in Calculus, can be interpreted as the direction or rate of the fastest increase in a given plot. Hence, it's inverse can be used to find a minimum of our Cost Function [3b1b]. So we will be tweaking our Theta Vector in the "direction" of the most profitable Thetas, i. e., the ones that minimize J. This method, however, is still vulnerable to be stuck at local minima, since there would be no direction in which the algorithm could "move" to lower the cost function. Furthermore, since this is a multiclass classification problem, we also need to employ the One Vs All strategy in this case, just like we did in Logistic Regression.

A Neural Network needs an activation function, in order to map results. Common choices are usually the Sigmoid (or Logistic Function) or Tanh, that map any real number to a number between 0 and 1, which proves to be extremely useful in classification problems, which is our case. Our activation function will be responsible for mapping the result of the Cost Function to a probability of outcome of each label. For instance, in the context of ASL, we will use the Sigmoid to achieve the vector of the model predictions for all training examples, which afterwards will be used to calculate the Cost Function and adapt Thetas accordingly.
Additionally, since we have a Classification problem with multiple labels, we will use the One versus All strategy, which essentially consists in mapping each of the desired outputs to a vector in which the index that corresponds to the desired label is 1 and the rest is 0. Lastly, with the resulting vector of costs, we will use the Softmax function, to get how likely a given label is to correspond to an example. [CITE LAST PAPER]

Just like in other models, we defined three different sets of hyper-parameters, to increase the chances of finding a well representative model of a Neural Network's performance in the context of this problem. The variation of the cost function of each developed model can be seen on the plot on figure X. Naturally, let us pick the set of hyper-parameters that converges to the lowest cost function the fastest. After validation, we observed a total accuracy of X %.


### Convolutional Neural Networks

A common variation of a Neural Network consists of a Convolutional Neural Network (CNN). In our project, a simple CNN was developed and used, with the help of libraries such as Tensorflow and Keras. Convolutional Neural Networks are  neural networks that use convolution in place of general matrix multiplication in at least one of their layers. According to Keras: "The right tool for an image classification job is a convnet." [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]

In this project, we implemented a CNN using third-party libraries such as TensorFlow and Keras. Keras allows building a Neural Network using a sequential model. In other words, the network itself is built by putting together layers. In our model we used Conv2D, MaxPooling and Dropout, developing a similar model to the one used in previous papers. [CITE MYSELF]

Our Neural Network performed fairly well for the training data, achieving an overall training accuracy of 98.37%. However, when validating it, the accuracy fell short of 91% with a total score of 90.24%. We can naturally see this model is overfit for the data, but achieved a nevertheless good score.


### Decision Trees

In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. As the name goes, it uses a tree-like model of decisions. Though a commonly used tool in data mining for deriving a strategy to reach a particular goal, it is also widely used in machine learning. [https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052]

<TODO_WRITE_MORE_THEORY_ABOUT_DECISION_TREES_ORNOT>

In our project, a straight-forward Decision Tree Classifier was used, using the class provided by the scikit-learn library. This model allows us to select a few hyper-parameters, such as the criterion used for the split, the maximum depth of the generated tree and the mininum number of samples required to split an internal node. On a first approach, we trained our model with the "entropy" criterion, along with max_depth of 3 and a min_samples of 5.

This resulted in a validation accuracy of 8.16%, so we can naturally infer our model is most likely overfitting the training data. After researching and reading further documentation [https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use], we can conclude this might be due to the high feature over training examples ratio. A possible workaround for this problem would be either through feature selection or data augmentation. However, since we already have well-performing models, we chose to simply discard this option for fine-tuning.


## Fine-tuning the model

Bearing in mind these results, we opted to fine-tune the Support Vector Machine model, since it was the one which presented the best accuracy. However, instead of using a Holdout validation set like we did when choosing a model, we opted to implement K-fold cross validation for each hyperparameter.

There is a number of ways to perform model validation. The reason behind our choice lies vastly on the results discussed by Rodríguez J. and Lazaro J. [https://www.researchgate.net/publication/224085226_Sensitivity_Analysis_of_k-Fold_Cross_Validation_in_Prediction_Error_Estimation]. According to the paper, K-fold validation represents a good sensitive measure to properly validate models. This way we can be sure of the results obtain as we fine-tune our model using the aforementioned algorithm.

Our model hyper-parameter selection was hence done by generating a logarithmically spaced range of values for both C and gamma. C controls the tradeoff between a smooth decision boundary and classifying training points correctly [https://www.youtube.com/watch?v=joTa_FeMZ2s]. In other words, a larger C will mean a lower bias  but, in turn, result in a higer variance for our model. A lower C generally implies higher bias but less varying results. On the other hand, we also fine-tuned the gamma hyper-parameter, is the parameter of a Gaussian Kernel, which is used to handle non-linear classification. A common solution for non-linearly separable data when using Support Vector Machines is to raise the data to a higher dimension, so that it can be separated using hyperplanes. Usually for such cases an RBF kernel is used, but as we can see from the plots in Figure X, we obtained far better results using a linear one. 

The method for finding the best hyperparameters was done using the Grid Search algorithm. Grid-searching is the process of scanning the data to configure optimal parameters for a given model. Depending on the type of model utilized, certain parameters are necessary, such as, in our case, C and gamma. Since Grid-searching can be applied across machine learning to calculate the best parameters to use for any given model, such an algorithm is vastly used when it comes to fine-tuning machine learning models. It is important to note, however, that Grid-searching can be extremely computationally expensive and may take quite a long time to run, since it iterates through every parameter combination and stores a model for each combination. In other words, Grid Search will build a model for each parameter combination possible and test it [https://medium.com/@elutins/grid-searching-in-machine-learning-quick-explanation-and-python-implementation-550552200596]. 

Each of these combinations is then tested and validated against one of the K-fold training-validation sets. In our method we used K = 3, meaning this rotation was done three times for each combination of parameters. The average of results was then calculated and plotted, as we can see in Figure X.

[INSERT PLOTS FOR SVMs {linear, rbf} HERE]


## Results

### Without feature extraction


### With feature extraction using Eigenfaces


## Conclusions



