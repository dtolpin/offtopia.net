---
title: How Data Scientists Fail
subtitle: What can go wrong in a data science task
date: 2022-09-04T16:09:33+03:00
draft: False
---

I am going to job interviews, again. This time, a frequent
request is: "Tell us about a failed project". Of course, I never
fail as a data scientist, how could I? A data science task
involves a combination of domain knowledge and data, neither is
held or produced by me, and a question someone else wants an
answer to. All I do as a data scientist is encoding the domain
knowledge as a model, updating the model's latent variables
based on the data, and computing a quantitative answer to the
question. There are ways to [ensure adequacy of the model, check
convergence of inference, and express uncertainty of the
answer](http://www.stat.columbia.edu/~gelman/book/).
Just doing all these steps by the book ensures that there is
absolutely no way to fail. Consider the task of classifying
[hand-written digits](http://yann.lecun.com/exdb/mnist/) ---
although different models may have different accuracy, there is
no way to ‘fail’ as long as one does things as taught. Or is
there?

<!--more-->


Let us see what a failure is. A failure is not a wrong model
choice, or poor convergence of inference, or a mistake in
computing compatibility intervals. Those are manifestations of
incompetence rather than failure. A failure happens when the
data scientist does everything right, but still causes a
disaster, hopefully small and easy to recover from. Let me
argue that a failure can only happen if the data scientist makes
a decision based on hard to validate assumptions, and those
assumptions turn out to be too far from the reality.

But do data scientists make any ‘voluntary’ decisions at all?
Turns out they do. If the task is label assignment, then the
decision is the compromise between precision and recall. For
forecasting, the compromise is between forecast stability and
width of confidence interval. For clustering, one has to
balance, explicitly or implicitly, between the number of
clusters and similarity of members of each cluster. Despite
apparent dissimilarities, all of these decisions are kinds of
[_exploration-exploitation
compromise_](https://towardsdatascience.com/the-exploration-exploitation-dilemma-f5622fbe1e82).  Exploration-exploitation
compromise always addresses yet unseen data and yet undiscovered
knowledge, and thus acting by the book does not guarantee
success. Sometimes, a wrong exploration-exploitation compromise
is made, and this is how a data science project fails.

To conclude, an example. I took upon a task of automated traffic
acquisition --- paying for visits to a web page to earn from
advertisements on that page. Visitors are acquired through an
auction, so one wants to bid higher if one anticipates higher
earnings. I deployed a model for temporal forecasting of visit
value, and a decision algorithm to choose the optimal bid given
the forecast. The algorithm accounted for forecast
uncertainty, maximized expected gain, took care of risks, and
did everything ‘right’, by the book. It worked well for a while,
but eventually --- and suddenly --- two different extreme cases
popped up, incurring losses (which luckily where quickly
mitigated):

* On a small number of campaigns, the actual visit value
  suddenly dropped after steadily going up, violating smoothness
  assumptions. A smoothness assumption is over-exploitation. The
  result was trading at loss, for a short time but with high
  traffic volume and cost.
* On another small group of campaigns, the traffic went down
  almost to zero due to a low visit value forecast, followed by
  low bids due to a broad safety margin. A broad safety margin
  is over-exploration, but with close to zero traffic
  the forecasting ceased to be reliable, resulting in wasted
  resources and lost opportunities. 

Both failures were fixed, eventually. What is important though
is the cause of the failures: both happened due to inadequate
exploration-exploitation assumptions introduced into the
algorithm, neither could be discovered based on either
historical data or model-based simulations.
