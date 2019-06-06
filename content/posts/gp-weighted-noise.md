---
title: "Gaussian process regression with varying noise"
subtitle: ""
date: 2019-06-06T11:00:00+03:00
draft: true
---

In Gaussian process regression for time series forecasting, all
observations are assumed to have the same noise. When this
assumption does not hold, the forecasting accuracy degrees.
Student's _t_-processes handle time series with varying noise
better than Gaussian processes, but may be less convenient in
application. In this article, we introduce a weighted noise
kernel for the Gaussian process allowing to account for varying
noise when the ratio between noise variances for different
points is known, such as in the case when an observation is the
sample mean of multiple samples, and the number of samples
varies between observations. A practical example of this setting
is forecasting of mean visitor value based on revenues and
numbers of visitors over fixed time intervals.

## Preliminaries

### Gaussian process regression

[Gaussian
process](https://en.m.wikipedia.org/wiki/Gaussian_process) is a
non-parameteric regression model in which the vector of values
of the target variable in any finite combination of points
follows the normal (Gaussian) distribution.  Formally, a
Gaussian process defines a distribution over functions:

$$f \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$$

Here, $m(\cdot)$ is called the _mean function_, and $k(\cdot,
\cdot)$ the _kernel_. Given any vector of points $\pmb{x}$,
the distribution of the function values at these points
follows multivariate normal distribution:

$$f(\pmb{x}) \sim \mathcal{N}(m(\pmb{x}), k(\pmb{x}, \pmb{x}))$$


Posterior inference is performed by computing the mean and
standard deviation at each point of interest based on values of
the target variable at the observed points. 

Inference depends on the process kernel. Kernels can be combined
by addition and multiplications, and most kernels are
parameterized by a small number of _hyperparameters_. The
hyperparameters are inferred ('tuned'), e.g. by maximizing the
likelihood of the training set.


### White noise kernel

To deal with noisy observations, a small constant $\sigma_n^2$
is customarily added to the diagonal of the covariance matrix
$\Sigma$:

$$\Sigma \gets \Sigma + \sigma_n^2I$$

The constant $\sigma_n^2$ is interpreted as the variance of
observation noise, normally distributed with zero mean. Instead
of adding the noise to the covariance matrix, a _white
noise kernel_ term can be added to the process kernel. The white
noise kernel $k_{n}(\cdot, \cdot)$ is specified as:

$$k_n(x, x') = \sigma_n^2 \text{ if } x \equiv x', 0 \mbox{ otherwise.}$$

Here, $\equiv$ means that $x$ and $x'$ refer to the same point,
rather than just to a pair of possibly different points with the
same coordinates.

## Weighted white noise kernel

### Learning the noise

### Forecasting

## Case study

For the case study, we obtained a time series where the
empirical mean is computed at each point, and the number
of samples varies in broad bound between points. Figure 1 
shows the empirical means and the numbers of samples. Obviously,
the empirical mean has higher variance at points with lower
numbers of samples.

![Empirical mean and numbers of samples](data.png)  
**Figure 1. Empirical means and numbers of samples.**

We implemented this 'weighted noise' trick for the
[scikit-learn](scikit-learn.org) version of Gaussian processes,
using the
[WhiteKernel](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html)
as the starting point and modifying the code to accept
observation weights. Figure 2 compares forecasting with fixed
(orange) and weighted (green) noise. The weighted noise
prediction gives much tighter confidence bounds, while still
closely following the dynamics of the average visit value.

![Visit value forecasting with weighted noise](/images/weighted-white/weighted-gp-forecast.png)  
**Figure 2. Forecasting with uniform and weighted noise.**

The implementation of the weighted white kernel for scikit-learn
used in the study is available at
[http://github.com/dtolpin/weighted-white-kernel](http://github.com/dtolpin/weighted-white-kernel).
