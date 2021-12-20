# Linear Regression {#chapter-3}

Chapter 3 is about (ordinary least squares and polynomial) linear regression, a parametric supervised statistical learning method. Many of the more advanced statistical learning approaches are generalizations or extensions of linear regression, so having a good grasp on linear regression is important for understanding later methods in the book.

It also discusses K-nearest neighbours regression (KNN regression), a non-parametric supervised statistical learning method.

## Exercises

### Prerequisites {.unnumbered}

To access the data sets and functions used to complete the Chapter 3 exercises, load the following packages.


```r
library(ISLR2)
library(tidyverse)
# library(skimr)
# library(GGally)
# library(patchwork)
```

### Conceptual {.unnumbered}

::: exercise
Describe the null hypotheses to which the p-values given in Table 3.4 correspond. Explain what conclusions you can draw based on these p-values. Your explanation should be phrased in terms of sales, TV, radio, and newspaper, rather than in terms of the coefficients of the linear model.

| Term      | Coefficient | Std. Error | t-statistic | p-value  |
|-----------|-------------|------------|-------------|----------|
| intercept | 2.939       | 0.3119     | 9.420       | \< .001  |
| TV        | 0.046       | 0.0014     | 32.81       | \< .0001 |
| radio     | 0.189       | 0.0086     | 21.89       | \< .0001 |
| newspaper | -0.001      | 0.0059     | -0.18       | .8599    |

: Table 3.4

*Answer*.

The null hypotheses to which the p-values in Table 3.4 correspond is that the coefficient for each term is equal to zero, $\beta_i = 0$, which we use to test whether a term is associated with the response variable. We can use the p-values in the table to infer whether the coefficients are sufficiently far from zero that we can be confident they are non-zero, and thus associated with the response variable. 

Based on the p-values in Table 3.4, we can conclude that:

- A $1000 increase in the TV advertising budget is associated with an average increase in product sales by 46 units while holding the advertising budgets for radio and newspaper fixed.
- A $1000 increase in the radio advertising budget is associated with an average increase in product sales by 189 units while holding the advertising budgets for TV and newspaper fixed.
- A $1000 increase in the newspaper advertising budget is not associated with a change in product sales.

Data note: For the Advertising data sales are in thousands of units and advertising budgets in thousands of dollars.
:::

::: exercise
Carefully explain the differences between the KNN classifier and KNN regression methods.

*Answer*.

Given a value for $K$ and a test observation $x_0$, both the KNN classifier and KNN regression first identify the $K$ training observations that are closest to $x_0$, represented by $\mathcal N_0$. For the KNN classifier $x_0$ would be a discrete class, and for KNN regression $x_0$ would be a continuous value. The methods then diverge in what they do with $\mathcal N_0$.

The KNN classifier then estimates the conditional probability that the test observation $x_0$ belongs to class $j$ as the fraction of training observations in $\mathcal N_0$ whose response value $y_i$ equals $j$:

$$
\mathrm{Pr}(Y = j|X = x_0) =  \frac{1}{K} \sum_{i \in \mathcal N_0} I(y_i = j).
(\#eq:knn-classification)
$$

KNN regression, on the other hand, then estimates $f(x_0)$ using the average of all the training observations in $\mathcal N_0$:

$$
\hat f(x_0) = \frac{1}{K} \sum_{i \in {\mathcal N}_0} y_i.
(\#eq:knn-regression)
$$

As can be seen in Equations \@ref(eq:knn-classification) and \@ref(eq:knn-regression), there are two main differences between the KNN classifier and KNN regression methods:

- On the right side of the equation, the KNN classifier uses an indicator function $I(y_i = j)$ to determine whether a response value $y_i$ equals $j$ (if so then it equals 1, if not 0). These are summed to get the numerator of the conditional probability for each class (where the denominator is $K$). KNN regression, on the other hand, simply uses the response value $y_i$ as is, since it's a continuous value. These are also summed to get the numerator of the fraction (again with $K$ as the denominator), although it makes more sense to think of this as the average of all the training observations in $\mathcal N_0$. 
- On the left hand side of the equation, for the KNN classifier we estimate the conditional probability that a test observation $x_0$ belongs to a class $j$. We can then use this probability to assign $x_0$ to the class $j$ with the highest probability, if we so choose (although we don't have to, the probabilities are equally useful). For KNN regression we estimate the form of $f(x_0)$, which corresponds to our prediction point for our test observation $x_0$ since $\hat Y = \hat f(X)$.

Thus, the main difference between the two methods is that the KNN classifier method is used to predict what class an observation likely belongs to based on the classes of its $K$ nearest neighbours, whereas the KNN regression method is used to predict what value an observation will have on a response variable based on the response values of its $K$ nearest neighbours.
:::

::: exercise
Suppose we have a data set with five predictors, $X_1 =$ GPA, $X_2 =$ IQ, $X_3 =$ Level (1 for College and 0 for High School), $X_4 =$ Interaction between GPA and IQ, and $X_5 =$ Interaction between GPA and Level. The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get $\hat \beta_0 = 50$, $\hat \beta_1 = 20$, $\hat \beta_2 = 0.07$, $\hat \beta_3 = 35$, $\hat \beta_4 = 0.01$, $\hat \beta_5 = −10$.

(a) Which answer is correct, and why?

i. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates.

ii. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates.

iii. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates provided that the GPA is high enough.

iv. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates provided that the GPA is high enough.

*Answer*. The fourth description is correct. The coefficient for Level is positive, so when it is included in the model (i.e., when Level is not zero) then predictions for starting salary will be higher; however, because the coefficient for the interaction between GPA and level is negative, this will only be true provided that college graduate GPA is high enough. 

(b) Predict the salary of a college graduate with IQ of 110 and a GPA of 4.0.

*Answer*.

The predicted salary for this college graduate is $137,100.

$$
\begin{align}
\hat y &= 50 + 20(4) + 0.07(110) + 35(1) + 0.01(4*110) - 10(4*1) \\
       &= 50 + 80 + 7.7 + 35 + 4.4 - 40 \\
       &= 137.1
\end{align}
$$

(c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is very little evidence of an interaction effect. Justify your answer.

*Answer*. False. The size of the coefficient does not provide evidence for or against an interaction effect. We use the p-value associated with the coefficient to determine whether it is sufficiently far from zero that we can be confident it is non-zero, and thus associated with the response variable. If our estimates are precise enough we can find a small coefficient and be confident that it is non-zero. But we cannot make a decision about this based on the coefficient alone.
:::

::: exercise
I collect a set of data (n = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e. $Y = \beta_0 + \beta_1X + \beta_2X^2 + \beta_3X^3 + \epsilon$.

(a) Suppose that the true relationship between $X$ and $Y$ is linear,
i.e. $Y = \beta_0 + \beta_1X + \epsilon$. Consider the training residual sum of squares (RSS) for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

*Answer*.

(b) Answer (a) using test rather than training RSS.

*Answer*.

(c) Suppose that the true relationship between $X$ and $Y$ is not linear, but we don’t know how far it is from linear. Consider the training RSS for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

*Answer*.

(d) Answer (c) using test rather than training RSS.

*Answer*.

:::
