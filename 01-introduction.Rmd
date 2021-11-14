# Introduction {#chapter-1}

Chapter 1 provides a brief overview of statistical learning, which refers to a vast set of tools for understanding data. The tools of statistical learning can be classified as either *supervised* or *unsupervised*.

__Supervised statistical learning__ involves building a statistical model for predicting an *output* (the response) based on one or more *inputs* (the features). In a typical supervised learning setting we have access to a set of $p$ features $X_1, X_2, ... , X_p,$ measured on $n$ observations, and a response $Y$ also measured on those same $n$ observations. The goal is then to predict $Y$ using $X_1, X_2, ... , X_p$. The response $Y$ can be *quantitative* (continuous, numerical) or *qualitative* (categorical, non-numerical). When the goal is to predict a numerical value, we call this a *regression* problem. When the goal is to predict a non-numerical value we call this a *classification* problem. Supervised statistical learning can be used for *exploratory* or *confirmatory* data analysis.

__Unsupervised statistical learning__ involves cases where one or more *inputs* are used to learn about relationships and structure in the data in the absence of an output. In a typical unsupervised learning setting we (only) have access to a set of $p$ features $X_1, X_2, ... , X_p,$ measured on $n$ observations. The goal is then to discover interesting things about the measurements on $X_1, X_2, ... , X_p$. When the goal is to partition observations into subgroups based on (dis)similarity, we call this a *clustering* problem. Unsupervised statistical learning is typically used for exploratory data analysis; because it is not possible to check our work with unsupervised statistical learning (we don't know the true answer since the problem is unsupervised), it is not used for confirmatory data analysis.

## Notation and Simple Matrix Algebra

Chapter 1 also discusses the notation conventions used in the textbook, starting on Page 9. Briefly:

- $n$ represents the number of observations in our sample
- $p$ represents the number of variables available in our data set
- $x_{ij}$ represents the value of the $j$th variable for the $i$th observation, where $i = 1, 2, ..., n$ and $j = 1, 2, ..., p$.
- $\mathbf X$ represents an $n \times p$ matrix whose $(i,j)$th element is $x_{ij}$
- $x_i$ represents the rows of a matrix $\mathbf X$ where $x_i$ is a vector of length $p$ containing the $p$ variable measurements for the $i$th observation.
- $\mathbf x_j$ represents the columns of a matrix $\mathbf X$ where $\mathbf x_j$ is a vector of length $n$ containing $n$ observation measurements for the $j$th variable.
- $y_i$ represents the $i$th observation of the response variable.
- $\mathbf y$ represents a vector of length $n$ containing the set of all response variable measurements or predictions.

## Data Sets Used in Labs and Exercises

The book uses data sets from the `ISLR2` package (available on CRAN) and one data set included as part of the base R distribution for labs and exercises.

## Book Website

The website for An Introduction to Statistical Learning is located at <https://www.statlearning.org>. It contains a number of additional resources that may be useful.
