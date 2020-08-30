---
title: "BDA Model Too Tough for Stan"
subtitle: "Estimating the population of NY state from sample summaries"
date: 2020-08-29T18:00:54+03:00
draft: False
---

I taught a course on Bayesian data analysis, closely following
[the book by Andrew Gelman et
al.](http://www.stat.columbia.edu/~gelman/book/), but with the
twist of using probabilistic programming, either
[Stan](http://mc-stan.org/) or [Infergo](http://infergo.org/),
for all examples and exercises. However, it turned out that at
least one important problem in the book is beyond the
capabilities of Stan.


This case study is inspired by 
Section 7.6 in [Bayesian Data
Analysis](http://www.stat.columbia.edu/~gelman/book/),
originally a paper [published in 1983 by Ronald
Rubin](https://www.sciencedirect.com/science/article/pii/B978012121160850017X).
The original case study evaluated Bayesian inference on the
problem of estimation of total population of 804 municipalities
of New York state based on a sample of 100 municipalities. Two
samples were given, with different summary statistics, and
power-transformed normal model was fit to the data to make
predictions consistent among the samples. The authors of the
original case study apparently had access to the full data set
(populations of each of 100 municipalities in both samples). 
However, only summary description of the samples appears in the
publication: the mean, the standard deviation, and
the quantiles:

&nbsp;      | Population | Sample 1 | Sample 2 
------------|------------|----------|----------
**total**   |13,776,663  |1,966,745 |3,850,502 
**mean**    |17,135      |19,667    |38,505 
**sd**      |139,147     |142,218   |228,625 
**lowest**  |19          |164       |162 
**5%**      |336         |308       |315 
**25%**     |800         |891       |863 
**median**  |1,668       |2,081     |1,740 
**75%**     |5,050       |6,049     |5,239 
**95%**     |30,295      |25,130    |41,718 
**highest** |2,627,319   |1,424,815 |1809578

This is a common way to summarize a data sample, and
[Pandas](https://pandas.pydata.org/), a Python library for data
analysis, even has a built-in function which produces such
summary for data. Here, we show how such summary description can
be used to perform Bayesian inference, with the help of
[stochastic conditioning](https://arxiv.org/abs/2001.02656).

The sample summary can be divided into three parts:

* the sample size;
* the mean and the standard deviation;
* the quantiles.

The sample size tells us how much information we have about the
distribution. The mean and the standard deviation describe the
distribution _parametrically_ --- if we knew the formula
of the probability density (parameterized by mean and standard
deviation), we could substitute these statistics into the
formula to fully specify the distribution. Finally, the
quantiles approximate the distribution shape
_non-parametrically_ --- we can sample from each quantile
proportionally to the probability mass of the quantile to
approximate samples from the distribution.

The original case study with comparing normal and log-normal
models, and finally fit a truncated three-parameter
power-transformed normal distribution to the data, which helped
to reconcile conclusions based on each of the samples while
producing results consistent with the total population. Here, we
use a model with log-normal sampling distribution and
normal-inverse-Gamma prior on the mean and variance. To complete
the model, we stochastically condition the model on the
piecewise-uniform distribution according to the quantiles:
$$z_{1\ldots\mathrm{n}} \leftarrow \mathrm{Quantiles}$$
$$m \sim \mathrm{Normal}(\mathrm{mean}, \frac {\mathrm{sd}} {\sqrt{\mathrm{n}}}),\quad s^2 \sim \mathrm{InvGamma}(\frac {\mathrm{n}} 2, \frac {\mathrm{n}} 2 \mathrm{sd}^2)$$
$$\sigma = \sqrt{\log \left(\frac {s^2} {m^2} + 1\right)},\quad\mu  = \log m - \frac {\sigma^2} 2$$
$$z_{1\ldots\mathrm{n}} \sim \mathrm{LogNormal}(\mu, \sigma)$$
By $z_{1\ldots\mathrm{n}}$ we denote $\mathrm{n}$ samples of
$z$ from the distribution defined by the quantiles. Here is the
model in Infergo:

{{<highlight go>}}
func (m *Model) Observe(x []float64) float64 {
	mean := x[0]
	vari := math.Exp(x[1])
	logp := x[1] +
		Normal.Logp(m.Mean, math.Sqrt(m.Vari/m.N), mean) +
		Gamma.Logp(m.N/2, m.N/2*m.Vari, 1/vari)

	sigma := math.Sqrt(math.Log(vari/(mean*mean) + 1))
	mu := math.Log(mean) - 0.5*sigma*sigma
	for i := 0.; i != m.N; i++ {
		z := <-m.Z
		logp += -math.Log(z) + Normal.Logp(mu, sigma, math.Log(z))
	}
	return logp
}
{{</highlight>}}

A straightforward way to sample from the quantiles is to sample a
quantile proportionally to the probability mass, and then sample
a value uniformly from the range of values in the quantile:

{{<highlight go>}}
func RandQuantile(q [][2]float64) float64 {
var p, z float64
for {
	p = rand.ExpFloat64()
	if p < 1 {
		break
	}
}
for i := 1; i != len(q); i++ {
	if q[i][0] >= p {
		z = q[i-1][1] + rand.Float64()*(q[i][1]-q[i-1][1])
		break
	}
}
return z
}
{{</highlight>}}

This is all we need to define the stochastically conditioned
probabilistic model in Infergo (the complete code and data are
[on BitBucket](https://bitbucket.org/dtolpin/stochastic-conditioning)).
We fit the model using sgHMC. The posterior predictive
distributions from both samples are quite similar and consistent
with the summary of the total population:

&nbsp;	  | Sample 1 | Sample 2
----------|----------|----------
**mean**  |18,646    |23,655
**5%**    |82        |69
**median**|2,389     |2,395
**95**    |66,381    |80,296


![Predictive posteriors](/images/nypopu/posteriors.svg)

The model can be improved  by replacing log-normal
with power-transformed normal distribution. However, the point of
this case study has been to show how combining parametric and
non-parametric summaries can be easily expressed with stochastic
conditioning. It is not clear to us how to express a
probabilistic program for this study otherwise, using
deterministic conditioning only.
