---
title: "How to Hug a Data Scientist"
date: 2018-04-09T14:19:13+03:00
draft: False
---

Sometimes, a data scientist is the first engineer in a software
project. More often though a data scientist joins the team when
there is working code, ready for deploying or even deployed.
Here is how the latter case rolls out:

> We write a piece of software. Thanks to continous delivery,
> we fix our bugs quickly and release new improved versions on
> time. Our code is fully tested, easy to change, and pieces
> fit each other smoothly.
> 
> Our concern is how smart our software is in making decisions
> and giving recommendations to the users. We figured out
> reasonable rules-of-thumb and built our algorithms around
> them. However, the quality of decisions degrades with time,
> and the rules need to be tweaked. Machine learning is a
> common buzzword for devising algorithms from collected data,
> isn’t it? We can use machine learning instead of manual
> tweaking. We bring a _data scientist_. The data scientist
> will add machine learning to our software.

![Hug](/images/hug.jpg)

Bringing a data scientist to the team may be tough — on both the
team and the data scientist. A simple reason is that the data
scientist may not be skilled enough in software development. But
let us assume that our data scientist understands how software
is built, is familiar with development methodologies, and can
write decent code. Still, there are things to do right.

Let me speak in the name of the data scientist. As a new team
member, I want us to succeed. To succeed for me means to
contribute to a product which makes decisions which are just
right for the user and stands ahead of competition by being both
smarter and more adaptive. For that, I need to understand the
big picture and identify decision points which may benefit from
machine learning algorithms. We will prioritize the points
together and choose a path of introducing machine learning to
the existing software system. I will then need to retrieve the
data, try a few approaches, and come up with algorithmic
solutions for the problems we identified — one-by-one. Finally,
we will connect those algorithmic solutions to acting
application code.  

## Stating the problems to solve

Asking the right questions and formulating the problems to solve
is as important as choosing the right algorithm. Let us, the
**architect** and the data scientist, to go over the system’s
architecture and pair-program a prototype of modifications
benefiting from machine learning. We can code the prototype in a
Jupyter notebook, or a similar friendly and dynamic environment.
Let me, the data scientist, lead the pair programming game. The
architect will bring to the table knowledge of the system, right
assumptions on data bandwidth and availability, and intuition
about impact of each possible improvement.

The code we’ll write together will go into trash. But quoting
Fred Brooks “The management question, therefore, is not whether
to build a pilot system and throw it away. You will do that.”
_(The Mythical Man-Month: Essays on Software Engineering, 1975)_.
The earlier we spend time writing throw-away code the better,
and the mutual understanding and communication we build along
the way will help me be productive later on.  

## Accessing the data

Finding, retrieving, and manipulating the data may be quite
frustrating. It is possible that I cannot get some of the data I
need, or cannot figure out how to access the data efficiently.
The best way for me — and for the project — is to pair-program
data access with the **data engineer** or a developer who
understands well the data interfaces and the underlying data
collection and storage architecture.

Let us co-author my data access code. Together, we will overcome
technical obstacles easier. And along the way, I will pick up
informal coding practices of the team and get to know the team
members better.  

## Behind-the-scenes pipeline

Machine learning brings to any software team, even the smallest
one, an aspect which is otherwise a property of large projects —
an **auxiliary pipeline**. Training, evaluation, and selection of
models dictates writing code which never ‘goes into production’.
It is tempting to be lax about quality of this part of the code
base. However, bugs in offline machine learning code may be as
dangerous as in the ‘core’ software: an inadequately trained
machine learning model will have adverse effect on the system’s
performance even if the online code is rigorously tested.

Let us develop the behind-the-scenes machine learning pipeline
together. Just like any other part of the software system we are
building, we must use a consistent development methodology, the
same methodology as for the rest of the system’s code.

## Connecting the wires

Output of a machine-learning based algorithm must ultimately be
turned into actions of the software system. This is where my
work leaves the realm of machine learning and enters the realm
of real-world decision making.

Collaboration and tight interaction are again important here.
Even if a well thought-out API exists for interfacing machine
learning with decision making, let us — me and a software
engineer responsible for the relevant part of the system —
pair-program the connection, with the software engineer leading
the effort. We will resolve discrepancies along the way, and I
will pass forward informal knowledge about the machine-learning
machinery I crafted. Later when a bug arises, for both of us
will be easier to pinpoint a likely cause and work out a
solution.

***

These are just a few highlights; there are certainly more
challenges in adopting machine learning technology in a software
project. To summarize, a condition of success is integration of
the data scientist into the engineering team. Every involved
team member should share with the data scientist responsibility
for the quality and efficiency of machine learning algorithms as
much as the data scientist shares with the rest of the team
responsibility for integrity and quality of the whole software
system.
