#  Random Forests

<p align="center">
  <img width="500" height="500" src="images/rf_preview.jpg">
</p>

## Goal

This project is part of my series of coding up some of the more classical machine learning algorithms from scratch! I find implementing these commonly used ML algorithms is really useful for understanding the ins-and-outs of how they really work, from input data to prediction output. 

The goal of this particular project is to code up the random forest algorithm from scratch with accuracy comparable to sklearn. 

As an aside, this from-scratch implementation includes support for out-of-bag (OOB) validation error estimation. 


## Description

Classification and regression decision trees excel at fitting a model to training data. However, decision trees have a tendency to overfit, meaning that they may not generalize well to previously-unseen test data. To improve generality and combat overfitting, random forests use a collection of decision trees that have been weakened to make them more independent. Essentially, we are trading a bit of accuracy for a much greater improvement to generality. 

Random forests don't feed all data to every decision tree in its collection due to a technique known as bootstrapping, which involves resampling the data with replacement. Each tree is trained on a bootstrapped subset of the original training data. To increase independence further, RFs can occasionally drop some of the available features during training. 

In this implementation, decision node splitting will be limited to considering a random selection of features of size `max_features`. Naturally, both bootstrapping and setting a maximum features per split will introduce noise into the predictions of the individual decision trees. But, averaging the results of these weakened tree estimators squeezes the noise back down, giving us close to the best of both worlds!

### Bootstrapping

As briefly mentioned above, the purpose of bootstrapping for random forests is to train a number of decision trees that are as independent and identically distributed as possible by using different but similar training sets.  Each tree trains on a slightly different subset of the training data. Bootstrapping, in theory, pulls from the underlying distribution that generated the data to generate another independent sample. In practice, bootstrapping about 2/3 of the training data and leaving 1/3 for "out of bag" (OOB) validation is a good rule of thumb or starting point. 

The algorithm for fitting a random forest is below:

<img src="images/fit.png" width="50%">

### RF Fitting

I went with a recursive approach in this implementation, which looks like:

<img src="images/dtreefit.png" width="50%">

For fitting conventional decision trees, `bestsplit()` exhaustively scans all available features and the feature values looking for the optimal variable/split combination. Optimal in this case depends on whether we are looking at a classification or regression problem. For classification, minimizing gini impurity or entropy are typical approaches. For regression, MSE is one such classic choice. This implementation allows the user to specify whatever loss function they'd like.

To reduce overfitting and promote independence amongst trees, each split should pick from a random subset of the features; the actual subset size is the hyperparameter `max_features`.  

<img src="images/bestsplit.png" width="60%">

### RF Prediction

Once we have our trained forest of decision trees, we can make predictions using `predict()`. For regression, the forest's prediction is simply the weighted average of the predictions from each individual decision trees. If `X_test` passed to `predict()` is a 2-D matrix of *n* rows, then *n* predictions will be returned as an array from `predict()`. To make a prediction for a single feature vector, call `leaf()` on each tree to get the leaf node that contains the prediction information. Each leaf has `n`, the number of observations in that leaf, which serves as our weight. The leaf also has a `prediction` that is the predicted y value for regression or class for classification. 

For classification, we need a majority vote across all trees.  As with regression, this implementation will sweep through all of the trees, and get the leaves associated with the prediction of a single feature vector. We then take the majority vote amongst all of leaf class predictions. Below is the prediction algorithm for regression and classification, respectively. 

<img src="images/predict-regr.png" width="60%">

<img src="images/predict-class.png" width="40%">

###  Regressor and classifier class definitions

To mimic sklearn machine learning models, I created some class definitions. 

The `RandomForest` class has my generic `fit()` method that is inherited by subclasses `RandomForest Regressor` and `RandomForestClassifier`.  Field `n_estimators` is the number of trees in the forest.

Method `compute_oob_score()` is a helper method used to encapsulate OOB validation score functionality. `RandomForest.fit()` calls  `self.compute_oob_score()` and that method then calls the implementation either in regressor or classifier, depending on which object was created.

Below is a class-based UML diagram of rf.py:

<img src="images/uml_class_diagram.jpg" width="60%">

## Out-of-bag (OOB) error

The advantage of OOB estimation is that the R^2 and accuracy scores are an accurate estimate of the validation error, all without having to manually hold out a validation or test set. This is another major advantage of random forests.

Generally, a bootstrapped sample is roughly 2/3 of the training records for any given tree, which leaves 1/3 of the samples (OOB) as test set. After training each decision tree, I keep track of the OOB records in the tree.  After training all trees in `fit()`, I loop through the trees again and compute the OOB score, if hyperparameter `self.oob_score` is true. The score is then saved in `self.oob_score_` for either the RF regressor or classifier object. Here are the OOB algorithms for regression and classification, respectively.

<img src="images/oob-score-regr.png" width="60%">

<img src="images/oob-score-class.png" width="60%">

## Thanks for reading!

Hope this little readme was helpful! If you'd like to see more ML algorithms implemented from scratch or leave any feedback/comments, feel free to check out the rest of my <a href="ahashemiz.github.io" target="_blank">portfolio</a> or message me on <a href="https://linkedin.com/in/arman-hashemizadeh" target="_blank">linkedin</a>
