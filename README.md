# ﻿Classifying Movie Success  
Timothy Nguyen  
Machine Learning Final Project - CS 483  
***See report.docx for a detailed explanation***

# Introduction
In this project, I will be attempting to predict if a movie is successful. A movie’s success can be interpreted in various ways: number of awards, total revenue, and box office. But a basic guideline is that a movie is considered profitable if the revenue is twice the budget [1]. So, success will be a boolean feature that is true if the movie revenue exceeds twice the budget. I will be attempting to predict this value using several classification algorithms: logistic regression, K-nearest neighbors, decision tree, and random forest.

# Data Collection
While there are many movie datasets available, I wanted to start from scratch. So, I collected my own data from The Movie Database API (TMDB) [2]. The API was accessed using the python library tmdbsimple. The movies were collected by iterating by year using the discover method from the API. The dataset consists of 375,377 movies dating from 1884 to 2018.

# Data Cleaning
Before this raw dataset is used for classification, it must be modified so it can be suitable for analysis. Many features are missing from many movies and it must be handled. New features will be created to help analysis. And analyzing the data will determine which features are useful for classification.

In the raw dataset, over 75% of movies have a missing budget and revenue. And about 100,000 movies have a null runtime. All these features are necessary for analysis, so the movie is dropped if any of these features are missing.  For this classification to be useful in determining the success for new movies, movies released before 1970 are removed. Also, movies with a vote_count less than 5 are removed to avoid the bias of only a few voters. For example, if a movie has a vote_average of 2 but only has 1 voter, it may not accurately represent the score. A movie is also dropped if it is a pornographic film. This is because nature of these movies is extremely different than theatric movies.

# Classification Models
To predict a movie’s success, four classification algorithms are used: logistic regression, K-nearest neighbors, decision tree, and random forest. Each classification algorithm will be using the same target value and predictors. The target value is success, a boolean value that is true if the revenue exceeds twice the budget. The predictors are listed below:

* budget – total money required to produce movie
* runtime – total movie time in minutes
* year – release year
* vote_average – mean community score
* vote_count – number of votes attributing to vote_average
* genre – most significant genre
* country – most significant production country
* certification_US – movie rating that determines suitability by viewer age [3]

For the categorical predictors (genre, country, certification_US), dummy variables are created so the algorithm can process them.
For each algorithm, sklearn’s grid search is used to find the hyper-parameters with the best accuracy. The grid search also uses a k-fold cross-validation where k=10 to ensure that the entirety of the dataset is tested. The model accuracy is measured using the mean of the test scores from the cross-validation. The mean training score will also be shown to determine if the model is overfitted or underfitted

# Conclusion

|    Algorithms               |    Mean   Test Score    |    Mean   Training Score    |    Hyper-parameters                                          |
|-----------------------------|-------------------------|-----------------------------|--------------------------------------------------------------|
|    Logistic   Regression    |    0.7047               |    0.7055                   |    solver:   newton-cg   C:   1000                           |
|    KNN                      |    0.6557               |    0.7251                   |    n-neighbors: 11                                           |
|    Decision   Tree          |    0.6581               |    0.7441                   |    criterion:   entropy,   max_depth:   6                    |
|    Random   Forest          |    0.6735               |    0.7558                   |    criterion:   gini   n_estimators:   9   max_depth:   8    |

The logistic regression algorithm yielded the best accuracy with a mean test score of 0.7047. It also has the lowest difference between mean test score and mean training score (0.0008), which means that it is fit neither overfitted nor underfitted.

This score is good, but not ideal. It should be higher to be able to confidently predict a movie’s success. I could have used more classification algorithms, but there was not enough time. For example, using SVM took about a day to calculate, and that is not including hyper-parameter tuning. I could have also cleaned up the data better during preprocessing and possibly removed some outliers. But the final dataset was small enough as it is, so reducing it would make the dataset useless in generalizing all movies.

# References
[1] https://io9.gizmodo.com/5747305/how-much-money-does-a-movie-need-to-make-to-be-profitable  
[2] https://developers.themoviedb.org/3/getting-started/introduction  
[3] https://www.mpaa.org/film-ratings/  
