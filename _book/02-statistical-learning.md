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
