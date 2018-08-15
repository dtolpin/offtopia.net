---
title: "A Small Program Can be a Big Challenge"
date: 2018-08-15T22:49:53+03:00
draft: true
---

A good part of today's internet content is created and shaped for delivering
advertisements. Internet articles are split into pages stitched by forward
links, so that the visitor reads the article in multiple steps in a fixed
order, and advertisements can be shown on every page. In a lucky case
for the publisher, the visitor clicks through the full article, eventually
reaching the last page. Most visitors though leave in the middle of an article,
either due to clicking on an advertising or just because they get bored 
and switch to other content or activity.

The distribution of the number of pages per visit is an important
metric for the publisher. It is used to both estimate revenues
from the advertising campaign and to optimize the article: order
of pages, page content, and advertisements appearing on each
page. Pages per visit constitute a counting time series.
There are established techniques for forecasting in counting time
series~\cite{TSC12}, however those techniques are mostly based on assumption
that the time series realizes a unimodal distribution at
every point in time, such as the Poisson or the Geometric
distribution. This assumption is inadequate for pages-per-visit
time series: there are usually several points in the article
where the visitor is very likely to leave; this commands a
multi-modal predictive distribution.

A sequence of Beta-Bernoulli distributions, one for each page,
gives a reasonable generative model for the number of pages per
visit. There are of course dependencies between pages --- a user
which is likely to leave on a certain page is also likely to
leave on `similar' pages, --- but we can ignore these
dependencies in an initial approximation. One phenomenon
should be accounted for though --- content managers
occasionally change the order and content of pages. The
posterior distribution should incorporate uncertainty due
to such changes.

## The Generative Model of a Visit

We view an article as a sequence of pages of length $N$.
A Beta-Bernoulli distribution $Beta\mbox{-}Bernoulli(\alpha_i, \beta_i)$
for the $i$th page for each $i \in \{1, \ldots, N\}$ models the
event of leaving after visiting the page. (The probability of leaving
after the last page is 1 here, however in reality the visitor 
can occasionally continue to another article on the same site, which should
be taken into account.)  We draw the number of pages per visit
by passing through the sequence until the `left' event (Algorithm~\ref{alg:drawing}).
\begin{algorithm}
    \caption{Drawing the number of pages per visit}
    \label{alg:drawing}
    \begin{algorithmic}
        \FOR {$i = 1$ \textbf{to} $N$}
           \STATE $left \sim Beta\mbox{-}Bernoulli(\alpha_i, \beta_i)$
           \IF {$left$}
              \RETURN $i$
           \ENDIF
        \ENDFOR
        \RETURN $N$
    \end{algorithmic}
\end{algorithm}

To update the beliefs based on the observed number of pages $K$, we
just increment $\beta_i$ if the visitor continued to the next
page, that is $i < K$, or $\alpha_i$ if the visitor left at the
$i$th page ($i = K$). We must also account for the uncertainty 
due to content changes (we are not informed about the changes).
After trying a few alternatives, we represent uncertainty as a
cap $C$ on the amount of evidence which 
we collect for each page. If $\alpha_i + \beta_i$ exceeds $C$ after
updating, we scale down both $\alpha_i$ and $\beta_i$ by the
same factor, such as the probability of leaving remains the same,
but $\alpha_i + \beta_i$ is equal to $C$
(Algorithm~\ref{alg:updating}).

\begin{algorithm}
    \caption{Updating the beliefs based on the observed number
    of pages per visit}
    \label{alg:updating}
    \begin{algorithmic}
    \FOR {$i = 1$ \textbf{to} $K$}
      \IF {$i < K$}
         \STATE $\beta_i \gets \beta_i + 1$
      \ELSE 
         \STATE $\alpha_i \gets \alpha_i + 1$
      \ENDIF
      \IF {$\alpha_i + \beta_i > C$}
         \STATE $\alpha_i \gets \alpha_i \cdot \frac C {\alpha_i + \beta_i}$
         \STATE $\beta_i \gets \beta_i \cdot \frac C {\alpha_i + \beta_i}$
      \ENDIF
    \ENDFOR
    \end{algorithmic}
\end{algorithm}

Obviously, we do not know the value of $C$ and want to infer 
$C$ for each campaign based on observed pages-per-visit counts.
We turn to probabilistic programming for this task, putting a
prior on $C$ and running the inference on the history of
pages-per-visit observations.

## Implementations

### Prototype in Anglican

For prototyping, we used Anglican~\cite{WVM14,TMY+16}. The model is straightforward
to implement in Anglican, a Lisp dialect. Since we only have a
single random variable to infer, Metropolis-Hastings Monte Carlo
performs well, and Anglican runtime is fast enough to produce
10\,000 samples in 45 seconds, given 100 observations.  However,
there were reasons that prevented us from using Anglican in
production, and we turned to other probabilistic programming
environments.

Namely, we looked for Python API, scalability, and support for
efficient variational inference~\cite{KTR+17}; Edward~\cite{THS+17} and
Pyro~\cite{Pyro18} were two obvious candidates.

### Implementation in Edward

Edward is based on TensorFlow~\cite{ABC+16} and uses TensorFlow API for
expressing flow control and variable updating. This results in
more verbose code which is hard to read and debug. However,
even when the challenge of expressing belief updates through
tensor manipulations is overcome, an additional difficulty
arises due to the complete separation between the generative
model and the data in Edward. Our probabilistic program both
conditions the distribution of $C$ and updates
beliefs of leaving the article at each page based on the data.
Specifying the model in a data-agnostic way is possible but
inference would becomes unreasonably inefficient.
In our simple case, belief updating and drawing the number of
pages per visit can luckily be disentangled --- the probability
of leaving the campaign at each page does not depend on the
updated beliefs for earlier pages. However, we still need to pass
the data twice --- both to the model and to inference.

Edward supports both Metropolis-Hastings and variational
inference. Metropolis-Hastings gave results consistent with the
Anglican implementation. One would expect static graph,
C++-based implementation of Metropolis-Hastings to run much
faster than in Anglican, however due to complex code having
to go through tensor manipulations, the performance was quite
poor --- Edward draws 10\,000 samples in 350 seconds, more than
7 times slower than Anglican. At the time of writing, the
implementation of variational inference in Edward has a
limitation preventing its application to our model.

### Implementation in Pyro

Pyro lets writing probabilistic programs in regular Python,
almost without limitation. Python is definitely sufficient for
implementing our probabilistic program. For inference, Pyro
supports importance sampling and variational inference, along
with other approaches.

We first run the inference with importance sampling, which gave
acceptable results, partially because our prior on $C$ was close
to the posterior. However, the running times were even longer
than with Edward: it takes more than 10 \textit{minutes} for Pyro to draw
10\,000 samples. We then turned to variational inference, only
to discover that the model would have to be rewritten: since gradients
are computed by the underlying PyTorch code~\cite{PGC+17},
all involved computations must be expressed as non-destructive
tensor manipulations, in a way similar to Edward implementation.
Variational inference gave results consistent with importance
sampling, however the learning rate had to be set low enough to
ensure convergence. Consequently, the running time of
variational inference was longer than of importance sampling for
similar accuracy.

## Round-up

For deployment in production, we implemented a custom
Metropolis-Hastings sampler, which was just a dozen lines in
Python. The performance is comparable to that of Anglican
implementation. However, this impedes our ability to extend the
probabilistic model and scale to larger amounts of data.

This case study points at properties of probabilistic
programming systems which are crucial for adoption but 
missing in many of the implementations.

*  Data structures must transparently support persistent
  updating and manipulation along with high-performance
    computation~\cite{O98}.  
*  Automatic differentiation algorithms which
  work well for deep learning are not necessarily good enough
  for probabilistic programming.
*  Small programs must run with acceptable performance. 
