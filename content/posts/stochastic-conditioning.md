---
title: "Stochastic conditioning"
subtitle: https://icml.cc/virtual/2021/spotlight/8608 
date: 2021-08-09T15:17:00+03:00
draft: False
---

Probabilistic programs implement statistical models. Commonly,
probabilistic programs follow the Bayesian generative pattern:

\begin{equation}
	\begin{aligned}
		x & \sim \mathrm{Prior} \\\\
		y & \sim \mathrm{Conditonal}(x)
	\end{aligned}
\end{equation}

* A prior is imposed on the latent variable $x$. 
* Then, observations $y$ are drawn from a distribution conditioned
on $x$.

The program and the observations are passed to an inference
algorithm which infers the posterior of latent variable $x$.

The questions is: what is observed?

In a basic, familiar, setting, samples from the joint data
distribution are observed. However, it is also possible that the
observations are

* samples from marginal distributions;
* summary statistics;
* and even the data distribution as a lazy infinite sampler. 

These cases frequently arise in real-life scenarios. Consider,
for example

* an **anonymized clinical study in a hospital**, where patients
bearing a particular disease are monitored, but only summary
statistics of each symptom are reported, to preserve patients'
privacy;

* or the famous **‘canadian traveller’ problem**, where the traveller
plans a road trip, but some roads can be closed due to bad
weather, so the model must be conditioned on the distribution
of possible road closure combinations.

Existing probabilistic programming frameworks cannot represent
such cases straighforwardly.

To address this problem, we propose to generalize deterministic
conditioning $x|y=y_0$ on values to stochastic conditioning $x|y
\sim D_0$ on distributions.

Formally, a probabilistic program with stochastic conditioning
is a tuple $(p(x, y), D)$, consisting of

* the computation $p(x, y) = p(x)p(y|x)$ of joint probability of assignments to x and y,
* and the observed distribution of D, with density $q(y)$. 

We aim to infer $p(x|y \sim D)$, the distribution of x given y
distributed as D. For that we need (unnormalized)

$$p(x, y \sim D) = p(x)p(y \sim D|x)$$

However, the model computes the joint probability $p(x, y)$
of x and y rather than of x and D. We need to express the
conditional probability of D given x in terms of y given x, and
D.  Thus, we define the former in terms of the latter:

$$p(y \sim D|x) \propto \exp \left( \int_Y (\log p(y|x))\,q(y)dy \right)=\prod\nolimits_Y p(y|x)^{q(y)dy}$$

> The probability of y distributed as D given x is the integral
> of log probability of y given x over D, exponentiated.
> Intuitively, this can be interperted as the product
> probability of all values of y given x, according to their
> observation probabilities.

Inference algorithms for deterministic conditioning cannot be
directly applied to programs with stochastic conditioning.
In principle, nested Monte Carlo estimation can be used,
but it is slow and cumbersome to code.

However, with our definition of conditional probability an
unbiased Monte Carlo estimate of log probability is available:

$$\log p(y \sim D|x) \approx \frac 1 N \sum\nolimits_{i=1}^N \log p(y_i|x)$$

This estimate is similar to the estimate used for subsampling 
of tall data. So, submsampling algorithms, such as

* pseudo-marginal MH;
* stochastic gradient MCMC;
* stochastic VI

can be used, with little modification.

We demonstrate stochastic conditioning in action on a case study
based on Donald Rubin's paper from 1983. The task is to estimate 
the total population fo New York state (804 municipalities)
based on a sample of 100 municipalities (two samples are
considered):

|            |Population (N=804) | Sample 1 (n=100) | Sample 2 (n=100)|
|------------|-------------------|------------------|-----------------|
|total|13,776,663|1,966,745|3,850,502 |
|mean|17,135|19,667|38,505 |
|sd|139,147|142,218|228,625 |
|lowest|19|164|162 |
|5%|336|308|315 |
|25%|800|891|863 |
|median|1,668|2,081|1,740 |
|75%|5,050|6,049|5,239 |
|95%|30,295|25,130|41,718 |
|highest|2,627,319|1,424,815|1809578|
>
The original study had access to populations of each
of the 100 municipalities in the sample, but the paper reports
only the summary statistics --- the mean, the standard deviation,
and the quantiles. Can we still perform inference?

It turns out we can, we stochastic conditioning, and here is the
model:

\begin{align}
        & y_{1\ldots n} \sim \mathit{Quantiles} \\\\
		& --------------------------- \\\\
        & m \sim \mathrm{Normal}\left(\mathit{mean}, {\mathit{sd}}/ {\sqrt n}\right) \\\\
		&\log s^2\sim\mathrm{Uniform}(-\infty,\infty) \\\\
		& \\\\
        & \sigma = \sqrt{\log \left(s^2/m^2 + 1\right)} \\\\
		& \mu  = \log m - {\sigma^2} / 2 \\\\
		& \\\\
        & y_{1\ldots n}\vert m,s^2 \sim  \mathrm{LogNormal}(\mu, \sigma)
\end{align}

Below the rule,
we define a model as though the populations of each municipality
were observed. We impose a prior on the parameters, and then
observe the populations from log-Normal distribution. However,
instead of passing fixed observations of populations, we
observe, through samples, a piecewise uniform quantile
distribution (above the rule). The model is differentiable, so
we can use  stochastic gradient Markov Chain Monte Carlo for
inference. 

![Posterior](/images/stochastic-conditioning/nypopu-estimate.svg)

In the posterior, the 95% compatibility intervals (solid
rectangles) for each of the two samples cover the true total
(red vertical line) and are even tighter than reported by Rubin
(using power-transformed normal), despite being based on less
information. 

[The paper](http://proceedings.mlr.press/v139/tolpin21a.html)
gives rigorous theoretical threatment of stochastic
conditioning, along with intuitive explanations on didactic
examples, and several elaborated case studies. The case studies
are implemented using Infergo, a probabilistic programming
framework for Go, and are available in a public [git
repository](https://bitbucket.org/dtolpin/stochastic-conditioning).
