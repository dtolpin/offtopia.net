---
title: Better Model Degrades Performance 
subtitle: and pushes research forward
date: 2026-07-10T17:49:00+0300
draft: False
---

Back in the good old days of
[Yoom](https://mc.linkedin.com/company/yoomiverse) I had trained a
neural density estimator on human body poses, and [Michal
Heker](https://www.linkedin.com/in/michal-heker-739767102/),
[Sefy Kagarlitsky](https://www.linkedin.com/in/sefykagarlitsky/)
and yours truly uploaded an [arxiv
paper](https://arxiv.org/abs/2507.12138) about this. One trick
described in the paper is that for proper density estimation in
6D rotation space, one needs to "de-Gram-Schmidt" the training
data, similarly to the dequantization trick, but for points on a
lower-dimensional manifold. However, the adoption of this trick was not smooth.

<!--more-->

First version of the model was trained _without
de-Gram-Schmidtization,_ and had "holes" in it due to
dimensionality mismatch between the latent and the ambient space
--- certain poses were assigned an anomalously high loss and a
random gradient; but with some guards and workarounds our
engineers got it to work in the pipeline, and the model helped
motion-capture optimization. That was a promising start, but
both in theory and in practice a neural estimator that naïvely
tried to learn an invertible map between data concentrated on a
63-dimensional manifold and a 126-dimensional Euclidean latent
space invited trouble.

Adding pose-preserving noise to 6D vectors seemed to be the
right solution. The effective dimensionality of both the ambient
and the latent space matched now, the model was easier to train,
and the gradients became well-behaved everywhere. I trained a
few model variants with different  noise levels, evaluated on my own
simulations confirming that the holes went away, and asked
the engineers to test on real motion capture. Rather
confusingly, the evaluation was disappointing: no improvement
compared to the previous model overall, and degradation here and
there, in difficult poses. 

I had a vague feeling that what was wrong was not the model itself
but how it was used; the best way to confirm my feeling would
have been to see where the gradient drives the pose in failure
cases --- but the engineers had their own ways to evaluate the
pipeline and the model, and I didn't get the feedback I needed
and asked for. This happens to data scientists, unfortunately
much more often than is justified. Both I and the developers
were left frustrated, each blaming the other side but lacking
strong enough arguments.

It was not until I started working on a different, but related,
regularization signal to the pipeline  that I realized that for
optimization, a global pose prior assigning a probabiity density
to every possible pose is the wrong signal. The old
model's gradients in rare poses were _just noise_, and with
stochastic optimization the noise cancelled itself --- the prior
didn't affect optimization in low-density regions. With the new
model, the gradients got _better everywhere_, and drove
optimization away from the right poses given the data to more
probable poses of the data distribution on which the model
was trained.

But one doesn't want to optimize towards a more probable pose
overall, one wants to optimize towards a more probable pose
given what is observed. In other words, the right signal for
pose optimization is a conditional prior, one that depends on
observables but supplements what cannot be observed (inverse
kinematics is unidentifiable for a human body model, with many
possible rotation combinations throughout the kinematic tree
resulting in the same joint or landmark locations). Such
a conditional prior is both easy to train and easy to apply:
RealNVP can use conditioning rather straightforwardly,  and the
conditioning signal can be obtained through forward kinematics
--- given a pose estimate, obtain joint (or other keypoint)
locations, and use the locations for conditioning and optimizing
towards a new pose estimate, and so on, in the
expectation-maximization style. A conditional prior with
de-Gram-Schmidtization is the signal that drives optimization in
the desired direction.

-----

It is a good time now, when the problem is understood and a
solution has been found, to reflect on what made me go with
the global prior initially. There were two reasons: first, I was
_asked_ to train a pose prior by the engineering team (and the
engineering team was motivated by many others who went this way,
[VPoser](https://github.com/nghorbani/human_body_prior) being
one example), second, being a Bayesian statistician I 
subconsciously recognized a global prior as a desirable part of
any model. This may be right (at least according to some points of
view) for posterior inference, but for optimization you want the 
prior signal to drive towards the conditionally optimal solution
rather than to a global maximum of some distribution, regardless
of which distribution it is. _Lesson learned._
