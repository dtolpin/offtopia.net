---
title: "How To Train Your Program"
subtitle: "A probabilistic programming pattern for Bayesian learning from data"
date: 2021-08-10T12:00:00+03:00
draft: False
---

[arXiv](https://arxiv.org/abs/2105.03650) | [code](https://bitbucket.org/dtolpin/h2typ-studies)

The ultimate Bayesian approach to learning from data is embodied by
hierarchical models. In a hierarchical model,
each observation or a group of observations $y_i$ corresponding
to a single item in the data set is conditioned on a parameter
$\theta_i$, and all parameters are conditioned on a
hyperparameter $\tau$:
\begin{equation} 
	\begin{aligned}
	\tau & \sim H \\\\
	\theta_i & \sim D(\tau) \\\\
	y_i & \sim F(\theta_i)
	\end{aligned}
	\label{eqn:hier}
\end{equation}

A hierarchical model can be thought of as a way of inferring, or
‘learning’, the prior of each $\theta_i$ from all observations
in the data set. Consider the following example problem:
multiple boxes are randomly filled by $K$ marbles from a bag
containing a mixture of blue and white marbles. We are presented
by a few draws with replacement from each of the boxes, $y_{ij}$
being the $j$th draw from the $i$th box; our goal is to infer
the number of blue marbles $\theta_i$ in each box.  Intuitively,
since the boxes are filled from the same bag, the posterior
distribution of $\theta_i$ should account both for draws from
the $i$th box and, indirectly, for draws from all other boxes.
This is formalized by the following hierarchical model:
\begin{align}
	\begin{aligned}
		\tau & \sim \mathrm{Beta}(1, 1) \\\\
		\theta_i & \sim \mathrm{Binomial}(K, \tau) \\\\
		y_{ij} & \sim \mathrm{Bernoulli}\left(\frac {\theta_i} K\right)
	\end{aligned}
	\label{eqn:marbles}
\end{align}

The above model learns from the data in the sense that
inference for each box is influenced by draws from all boxes.
However, learning from _training data_ to improve
inference on _future data_ with a  hierarchical model is
computationally inefficient --- if a new box is presented, one
has to add observations of the new box to the previously
available data and re-run inference on the extended data set.
Inference performance can be improved by employing data
subsampling, but the whole
training data set still needs to be kept and made accessible to the
inference algorithm.  A hierarchical model cannot ‘compress’, or
summarize, training data for efficient inference on future
observations.  Is there a way to learn from data in probabilistic
programs which is both Bayesian in nature and computationally
efficient?

## Problem: Learning from Data

The challenge we tackle here is re-using inference outcomes on
the training data set for inference on new data. Formally,
population $\mathcal{Y}$ is a set of sets $y_i \in Y$ of
observations $y_{ij} \in y_i$.  Members of each $y_i$ are
assumed to be drawn from a known distribution $F$ with
unobserved parameter $\theta_i$, $y_{ij} \sim F(\theta_i)$.
$\theta_i$ are assumed to be drawn from a common distribution
$H$. Our goal is to devise a scheme that, given a subset $Y
\subset \mathcal{Y}$, the _training set_, infers the
posterior distribution of $\theta_k|Y, y_k$ for any $y_k \in
\mathcal{Y}$ in a shorter amortized time than running inference
on a hierarchical model $Y \cup \{y_k\}$.  By amortized time we
mean here average time per $y_k,\,k \in 1:K$ as $K \to \infty$.

In other words, we look for a scheme that works in two stages.
At the first stage, inference is performed on the training set
$Y$ only. At the second stage, the inference outcome of the
first stage is used, together with $y_k$, to infer $\theta_k|Y,
y_k$. We anticipate a scheme that ‘compresses’ the training set
at the first stage, resulting in a shorter running time of the
second stage. Such scheme bears similarity to the conventional
machine learning paradigm: an expensive computation on the
training data results in shorter running times on new data.

## Main Idea: Stump and Fungus

In quest of devising such a scheme, we make two
observations which eventually help us arrive at a satisfactory
solution:

* In Bayesian modelling, information about data 
	is usually conveyed through conditioning of the
	model on various aspects of the data.
* In a hierarchical model, influence of the $i$th group of
	observations on the hyperparameters $\tau$ and, consequently,
	on other groups, passes exclusively through the group parameters
	$\theta_i$.

![stump and fungus](/images/h2typ/tree-and-fungi.svg)

If, instead of conditioning on training data $y_i$, we could
condition on parameters $\theta_i$ corresponding to the training
data, then we could perform inference on new data item $y_k$ at
a lower time and space complexity.  Continuing the well known
analogy  between a hierarchical model and a tree, with the
hyperparameter $\tau$ at the root and observations ${y_i}_j$ in
the leaves, we can liken a model which receives all $\theta_i$
of the training data and new data item $y_k$ as a _stump_ (the
hierarchical model with the trunk cut off just after the
hyperparameters) and a _fungus_ growing on the stump ---
the new data item.  The problem is, of course, that we infer
**distributions**, rather than fixed values, of $\theta_i$,
and the model must be, somewhat unconventionally, conditioned on
the distributions of $\theta_i$.

However, a recently introduced notion of [stochastic
conditioning](http://proceedings.mlr.press/v139/tolpin21a.html)
makes conditioning on distributions of $\theta_i$ possible, both
theoretically and in the practical case when the posteriors of
$\theta_i$ are approximated using Monte Carlo samples. Moreover,
conditioning the model both _stochastically_ on the posterior
distributions of $\theta_i$ on training data and
_deterministically_ on new data $y_k$ yields the same posterior
distribution of $\theta_k$ as inference on the full hierarchical
model. Based on this, we propose the ‘stump-and-fungus’ pattern
for learning from data in probabilistic programs:

* Training is accomplished through inference on a
  hierarchical model, in the usual way. 
* Training outcomes are summarized as a collection of
  samples $\tilde \theta$, representing the
  mixture distribution of $\theta_i$ of all groups.
* For inference on new data item $y$, a stump-and-fungus
  model is employed:
\begin{equation}
	\begin{aligned}
		\tilde \theta & \sim \mathrm{Hierarchical}(Y) \\\\
		---&--------------------- \\\\
		\tau & \sim  H \\\\
		\tilde \theta, \theta &|\tau \sim D(\tau) \\\\
		y&|\theta \sim F(\theta)
	\end{aligned}
\end{equation}

Although two models  --- hierarchical and stump-and-fungus ---
are involved in the pattern,  the models are in fact two roles
fulfilled by the same generative model, combining stochastic
conditioning on training data and deterministic conditioning on
new data (consisting potentially of multiple data items).  This
preserves a common practice in machine learning in which the same
model is used for both training and inference.

## Case Study: Tumor Incidence in Rats

In this case study,  discussed in
Chapter 5 of [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/), data on tumor incidence in rats in $N=70$
laboratory experiments is used to infer tumor incidence based on
outcomes of yet another experiment. A different number of rats
$n_i$ was involved in each experiment, and the number of tumor
cases $y_i$ was reported.
The posteriors of independent models for all experiments are
shown in Figure 1.

![separate](/images/h2typ/separate.svg)

Figure 1. Posteriors of separate models for all experiments

A schoolbook solution for the problem
is to perform inference on a hierarchical model:
\begin{equation}
	\begin{aligned}
		\alpha, \beta & \sim \mathrm{Uniform}(0, \infty) \\\\
		p_i&|\alpha, \beta \sim \mathrm{Beta}(\alpha, \beta)
		\\\\
		y_i&|p_i \sim Binomial(n_i, p_i) 
	\end{aligned}
\end{equation}
Inference on this model can be performed
efficiently thanks to summarization of $n$ observations from
$\mathrm{Bernoulli}(p)$ as a single observation from
$\mathrm{Binomial}(n, p)$. In general however, the use of a
hierarchical model would require carrying all observations of
all previous experiments for learned inference on findings of a
new experiment. 

The stump-and-fungus pattern is straightforwardly applicable to
the problem:
\begin{equation}
	\begin{aligned}
		p_{-k} & \sim  \mathrm{Hierarchical}(n_{-k}, y_{-k}) \\\\
		---&------------------------ \\\\
		\alpha, \beta & \sim \mathrm{Uniform}(0, \infty) \\\\
		p&|\alpha, \beta \sim \mathrm{Beta}(\alpha, \beta) \\\\
		y_k&|p_k \sim \mathrm{Binomial}(n_k, p_k) 
	\end{aligned}
	\label{eqn:rats-stump-and-fungus}
\end{equation}
$p_{-k}$ are stochastically observable from the posterior of
the hierarchical model conditioned on all but
the $k$th experiment. Figure 2 shows the posterior distributions
for $p$ inferred on the hierarchical model and through $N+1$
applications of stump-and-fungus~(similar to leave-one-out
cross-validation).  The [Infergo](https://infergo.org/) source
code of the model is provided in the appendix.  The inference
was performed with HMC on the hierarchical model and _stochastic
gradient_ HMC on the stump-and-fungus model. 1000 samples were
used to visualize the posterior. One can see that the posteriors
obtained via either method appear to be the same, except for
small discrepancies apparently caused by finite sample size
approximation.  The data and code for the case study are
available on
[BitBucket](https://bitbucket.org/dtolpin/h2typ-studies}).


| Hierarchical model | Stump-and-Fungus model |
|--------------------|------------------------|
|![tree](/images/h2typ/tree.svg)|![fungi](/images/h2typ/fungi.svg)|

Figure 2: Hierarchical vs Stump-and-Fungus 

## Discussion

We presented a probabilistic programming pattern for Bayesian
learning from data. The importance of learning from data is well
appreciated in probabilistic programming. Along with empirical
Bayes, applicable to probabilistic programming as well as to
Bayesian generative models in general, probabilistic-programming
specific approaches were proposed.  Our approach to learning
from data in probabilistic programs does not require any
particular implementation of probabilistic programming to be
used, nor introspection into the structure of probabilistic
programs or inference algorithms.  Instead, the approach uses
inference in ubiquitously adopted hierarchical models for
training, and conditioning on observations for incorporation of
training outcomes in inference. 
