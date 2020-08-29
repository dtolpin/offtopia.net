---
title: "Fitting a BDA model Stan can't handle"
subtitle: "Estimating population of NY state from sample summaries"
date: 2020-08-29T18:00:54+03:00
draft: True
---

\subsection{Estimating the population of New York state}

This case study is inspired by~\cite{R83}, also appearing as
Section~7.6 in~\cite{GCS+13}. The original case study evaluated
Bayesian inference on the problem of estimating the total
population of 804 municipalities of New York state based on a
sample of 100 municipalities. Two samples were given, with
different summary statistics, and power-transformed normal model
was fit to the data to make predictions consistent among the
samples. The authors of the original case study apparently had
access to the full data set (populations of each of 100
municipalities in both samples). 

\begin{table}
	\centering
	\begin{tabular}{l r r r}
		& Population & Sample 1 & Sample 2 \\
		& (N=804) & (n=100) & (n=100) \\ \hline
mean     &17,135      &19,667    &38,505 \\
sd       &139,147     &142,218   &228,625 \\
lowest   &19         &164      &162 \\
5\%      &336        &308      &315 \\
25\%     &800        &891      &863 \\
median   &1,668       &2,081     &1,740 \\
75\%     &5,050       &6,049     &5,239 \\
95\%     &30,295      &25,130    &41,718 \\
highest  &2,627,319    &1,424,815  &1809578
	\end{tabular}
	\caption{Summary statistics for populations of
	municipalities in New York State in 1960 (New York City was
	represented by its five boroughs); all 804 municipalities
	and two independent simple random samples of 100.
	From~\cite{R83}.}
	\label{tab:nypopu-data}
\end{table}
However only summary description of the samples appears in the
publication: the mean, the standard deviation, and
the quantiles (Table~\ref{tab:nypopu-data}). We show how
such summary description can be
used to perform Bayesian inference, with the help of stochastic
conditioning.

The original case study in~\cite{R83} started with comparing
normal and log-normal models, and finally fit a truncated three-parameter
power-transformed normal distribution to the data, which helped
reconcile conclusions based on each of the samples while
producing results consistent with the total population. Here,
we use a model with log-normal sampling distribution and
normal-inverse-Gamma prior on the mean and variance. To complete
the model, we stochastically condition the model on the
piecewise-uniform distribution $D_z$ of municipality populations 
according to the quantiles:
\begin{equation}
\begin{aligned}
	z_{1\ldots\mathrm{n}} & \leftarrow \mathrm{Quantiles} \\ \midrule
	m & \sim \mathrm{Normal}(\mathrm{mean}, \frac {\mathrm{sd}} {\sqrt{\mathrm{n}}}), & s^2  \sim \mathrm{InvGamma}(\frac {\mathrm{n}} 2, \frac {\mathrm{n}} 2 \mathrm{sd}^2) \\
	\sigma & = \sqrt{\log \left(s^2/m^2 + 1\right)}, & \mu  = \log m - \frac {\sigma^2} 2 \\
	z_{1\ldots\mathrm{n}} & \sim  \mathrm{LogNormal}(\mu,
	\sigma) &
\end{aligned}
	\label{eqn:nypopu}
\end{equation}

\begin{figure}
    \includegraphics[width=\linewidth]{nypopu-posteriors}
	\caption{Populations of municipalities in NY state: inferred predictive
	posterior distributions.}
	\label{fig:nypopu-posteriors}
\end{figure}
Despite apparently different summary statistics of the two
samples, the posterior distributions of municipality populations
come out quite similar (Figure~\ref{fig:nypopu-posteriors})
and they are consistent with the total population.
The posterior predictive distributions from both samples are
shown in Figure~\ref{fig:nypopu-posteriors} and summarized 
in Table~\ref{tab:nypopu-posteriors}. Despite differences in the
sample summaries, the posteriors are quite similar and
consistent with the summary of the total population.
\begin{table}
	\centering
	\begin{tabular}{l r r}
		& Sample 1 & Sample 2 \\ \hline
mean     &18,646   &23,655 \\
5\%      &82       &69 \\
median   &2,389     &2,395 \\
95\%     &66,381    &80,296
	\end{tabular}
	\caption{Posterior intervals for each of the samples.
	Despite apparent differences in the summaries
	(Table~\ref{tab:nypopu-data}), the posterior intervals are
	similar.}
	\label{tab:nypopu-posteriors}
\end{table}


