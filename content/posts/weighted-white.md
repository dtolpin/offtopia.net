---
title: "There Are No Outliers"
subtitle: "Gaussian process regression with varying noise made easy"
date: 2019-06-03T17:39:28+03:00
draft: true
---

Gaussian processes are great for time series forecasting. The
time series does not have to be regular --- 'missing data' is
not an issue.  A kernel can be chosen to express trend,
seasonality, various degrees of smoothness, non-stationarity.
External predictors can be added as input dimensions. A prior
can be chosen to provide a reasonable forecast when little
or even no data is available.

The only thing a Gaussian process does not deal well with is
outliers. 

