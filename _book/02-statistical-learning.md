# Statistical Learning {#chapter-2}

Chapter 2 formalizes the concept of statistical learning by introducing the general statistical model used for modelling the relationship between $Y$ and $X = (X_1, X_2, \dots, X_p)$, which can be written as

$$
Y = f(X) + \epsilon.
(\#eq:gsm)
$$

Here:

- $Y$ represents the response variable in our data set
- $X$ represents the set of variables in our data set
- $X_p$ represents the $p$th variable in our data set
- $f(\dots)$ represents a fixed but unknown function of its input(s)
- $\epsilon$ represents a random error term which is independent of $X$ and has mean zero

The goal of statistical learning is to estimate $f$. There are two main reasons for estimating $f$: *prediction* and *inference*. Depending on whether our ultimate goal is prediction, inference, or some combination of the two, different methods for estimating $f$ may be appropriate. In general, there is a trade-off between prediction accuracy and model interpretability. Models that make more accurate predictions tend to be less interpretable, and models that are more interpretable tend to make less accurate predictions (although this is not always the case, due to the potential for *overfitting* in highly flexible models).

The methods we use to estimate $f$ can be characterized as either *parametric* or *non-parametric*. Parametric methods involve a two-step model-based approach: First we make an assumption about the functional form of $f$ (e.g., we could assume $f$ is linear). Second we *fit* (train) the model to our training data in order to estimate the parameters. Non-parametric methods do not make any assumptions about the functional form of $f$. Instead they try to find an estimate of $f$ that gets close to the data points without being too rough or wiggly.

## Prediction

Because the error term $\epsilon$ averages to zero, the general statistical model for predicting $Y$ from $X = (X_1, X_2, \dots, X_p)$ can be written as

$$
\hat Y = \hat f(X).
(\#eq:gsm-p)
$$

Here:

- $\hat Y$ represents the resulting prediction for $Y$
- $\hat f$ represents our estimate for $f$

When our goal is only to predict, we do not typically need to concern ourselves with the exact form of $\hat f$ provided that it accurately predicts $Y$. The accuracy of $\hat Y$ as a prediction for $Y$ depends on two sources of error: *reducible error* and *irreducible error*. The error in our model attributable to $\hat f$ is *reducible* because we can potentially improve the accuracy of $\hat f$ for estimating $f$ by using a more appropriate statistical learning technique. However, the error in our model attributable to $\epsilon$ is *irreducible* because $Y$ is also a function of $\epsilon$, and $\epsilon$ is independent of $X$, so no matter how well we estimate $f$, the variability associated with $\epsilon$ will still be present in our model. This variability may come from unmeasured variables that are useful for predicting $Y$, or from unmeasurable variation. Irreducible error places an (often unknowable) upper bound on the accuracy of our prediction for $Y$.

## Inference

When our goal is to understand the relationship between $Y$ and $X = (X_1, X_2, \dots, X_p)$, we do need to concern ourselves with the exact form of $\hat f$. Here the form of $\hat f$ can be used to identify:

- Which predictors are associated with the response
- the direction (positive or negative) and form (simple or complex) of the relationship between the response and each predictor

## Assessing Model Accuracy

No one statistical learning approach performs better than all other approaches on all possible data sets. Because of this, care needs to be taken to choose which approach to use for any given data set to produce the best results. A number of important concepts arise when selecting a statistical learning approach for a specific data set:

- Measuring the quality of fit
- The bias-variance trade-off

### In the Regression Setting

In the regression setting, the most commonly used quality of fit measure for training data is the mean squared error $\mathit{MSE}$, given by


$$
\mathit{MSE}_{\mathrm{training}} = \frac 1 n \sum_{i = 1}^n (y_i - \hat f(x_i))^2,
(\#eq:mse-training)
$$

where $\hat f(x_i)$ represents the prediction that $\hat f$ gives for the $i$th observation. When predicted responses are very close to true responses the $\mathit{MSE}$ will be small; When predicted responses are very far from true responses the $\mathit{MSE}$ will be large. We generally do not really care about this value because accurately predicting data we have already seen is not particularly useful.

when our goal is to assess the accuracy of predictions when we apply our method to previously unseen *test data*, we can compute the mean squared error for test observations, given by

$$
\mathit{MSE}_{\mathrm{testing}} = \frac 1 n \sum_{i = 1}^n (y_0 - \hat f(x_0))^2,
(\#eq:mse-testing)
$$

where $(x_0, y_0)$ is a previously unseen test observation not used to train the statistical learning model. We want to choose the model that gives the lowest *test* $\mathit{MSE}$ by minimizing the distance between $\hat f(x_0)$ and $y_0$. When a test data set is available then we can simply evaluate Equation \@ref(eq:mse-testing) and choose the statistical learning model where $\mathit{MSE}$ is the smallest. When a test data set is not available then we can use *cross-validation*, which is a method for estimating test $\mathit{MSE}$ using the training data set.

The expected test $\mathit{MSE}$ for a given value $x_0$ can always be decomposed into the sum of three fundamental quantities: the *variance* of $\hat f(x_0)$, the squared *bias* of $\hat f(x_0)$, and the variance of the error term $\epsilon$, written as

$$
E \bigl(y_0 - \hat f(x_0) \bigr)^2 = \mathrm{Var}(\hat f(x_0)) +
                                     [\mathrm{Bias(\hat f(x_0))}]^2 +
                                     \mathrm{Var}(\epsilon),
(\#eq:expected-test-mse)
$$

where the notation $E \bigl(y_0 - \hat f(x_0) \bigr)^2$ defines the expected test $\mathit{MSE}$ at $x_0$. The overall expected test $\mathit{MSE}$ is given by averaging $E \bigl(y_0 - \hat f(x_0) \bigr)^2$ over all possible values of $x_0$ in the test data set.

The variance of $\hat f$ refers to the amount by which $\hat f$ would change if we estimated it using a different training set. The bias of $\hat f$ refers to the error that is introduced by approximating a real-life problem with a much simpler model. In general, as models become more flexible, the variance will increase and the bias will decrease. The relative rate of change of the variance and bias determines whether the test $\mathit{MSE}$ increases or decreases. Because the variance and bias can change at different rates in different data sets, the challenge lies in finding a model for which both the variance and bias are lowest.

### In the Classification Setting

In the classification setting, the most commonly used quality of fit measure for training data is the *error rate*, given by

$$
\frac 1 n \sum_{i = 1}^n I(y_i \ne \hat y_i).
(\#eq:er-training)
$$

Here:

- $\hat y_i$ is the predicted class label for the $i$th observation using $\hat f$
- $I(y_i \ne \hat y_i)$ is an indicator variable that equals one if $y_i \ne \hat y_i$ (incorrectly classified) and zero if $y_i = \hat y_i$ (correctly classified)

The *test error rate* for a set of test observations $(x_0, y_0)$ is given by

$$
\frac 1 n \sum_{i = 1}^n I(y_0 \ne \hat y_0),
(\#eq:er-testing)
$$

where $\hat y_0$ is the predicted class label that results from applying the classifier to the test observation with predictor $x_0$.

The bias-variance trade-off is present in the classification setting too; when variance and bias are lowest then the test error rate will be at its smallest for a given data set.

## Exercises

### Conceptual {-}

::: {.exercise}
For each of parts (a) through (d), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.

(a) The sample size $n$ is extremely large, and the number of predictors $p$ is small.

*Answer*. Better. A flexible model will generally be able to better estimate the the true $f$ and avoid overfitting when we have an extremely large sample size and a small number of predictors. The only exception would be if the true $f$ is linear, then an inflexible model would generally perform better; however, most real world relationships are not linear, so the lower bias of a flexible model will generally lead to a better quality of fit.

(b) The number of predictors $p$ is extremely large, and the number of observations $n$ is small.

*Answer*. Worse. A flexible model will generally lead to overfitting of our training data when the number of predictors is large and the sample size is small. An inflexible model is less likely to lead to overfitting in this scenario, so it will generally do a better job of giving accurate predictions on new observations than the flexible but overfit model.

(c) The relationship between the predictors and response is highly non-linear.

*Answer*. Better. A flexible model will generally be able to fit a highly non-linear relationship better than an inflexible model because the relative rate of decrease in bias tends to be much greater than the relative increase in variance when $f$ is highly non-linear. The left and right plots in Figure 2.12 on Page 36 of the book demonstrate this nicely.

(d) The variance of the error terms, i.e. $\sigma^2 = \mathrm{Var}(\epsilon)$, is extremely high.

*Answer*. Worse. A flexible model will generally lead to overfitting of our training data when the variance of the error terms is extremely high. Because $Y$ is partly a function of $\epsilon$, when the variance of the error terms is extremely high then the variance of $Y$ will also be extremely high, mainly due to random error. A flexible model that tries to find patterns in this noise is more likely to pick up on patterns that are caused by random chance rather than true properties of the unknown function $f$. The bias of an inflexible model is preferable in this situation, as it will give more stable predictions in the long run, which is likely preferable to making the essentially random predictions a flexible model would give in these circumstances.
:::

::: {.exercise}
Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide $n$ and $p$.

(a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry, and the CEO salary. We are interested in understanding which factors affect CEO salary.

*Answer*. Regression, inference, $n = 500$, $p = 3$.

(b) We are considering launching a new product and wish to know whether it will be a *success* or *failure*. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

*Answer*. Classification, prediction, $n = 20$, $p = 13$.

(c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.

*Answer*. Regression, prediction, $n = 52$, $p = 3$.
:::

::: {.exercise}
We now revisit the bias-variance decomposition.

(a) Provide a sketch of a typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The $x$-axis should represent the amount of flexibility in the method, and the $y$-axis should represent the values for each curve. There should be five curves. Make sure to label each one.

*Answer*.

(b) Explain why each of the five curves has the shape displayed in part (a).

*Answer*.
:::
