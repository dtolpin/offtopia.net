---
title: Claude Slop
subtitle: Surgical team with iron men
date: 2026-07-06T12:32:37+0300
draft: True
---

Background

Claude code is a dumb program that does what an LLM tells it to do; a mechanical executor.  The LLM that drives Claude  code was trained on 1) linguistic correctness as an autoregressive text generator 2) human feedback  (what is called RLHF). For human feedback, it was not trained to do 'the right thing', there is no way to train like that. It was trained for confirmation bias. That is, whatever you ask claude code to do, LLM instructs claude to do that in such a way that you think the task was accomplished successfully. That means it is trained to 'summarize' the work done in a way that you do not feel the urge to look at the code or throroughly check the results, relying on claude code's  diagnostic output instead.

The result of the routine you describe
(We plan our scemes by claude
We write our code by claude
we run our programs via claude
We debug and evaluate our results by claude
We analyze the results we get by claude)
is what helps Anthropic sell more tokens. It does not help write working code. It produces slop.

My personal experience is whenever opus/gpt/another llm are allowed to 'check itself' it always produces slop.  Two examples:


1. The first version of the flow I showed yesterday was based on forward kinematics: the LLM predicted euler angles. It didn't work because LLM predicted angles in global axes but converted them to rotation matrices as though they are local. To fix that, LLM built a skill of rules in which pose which global angles applied as local angles produce which results, a huge skills. Kind of worked on simple examples, lots of code, lots of data, everything seems to be fine and progressing until stuck..
2. I needed an extension so that pi does not go through shell substitutes every time a crop tool is called. The LLM instructed to write the extension in typescript (of course), but embedded the python cropping code inside the extension, so that I could not test the crop tool outside of the framework, and could not use the crop tool with another agent (not pi.dev). .

There are more examples. claude that checks and guides itself comforts itself into total very convincingly looking bullshit.

Programmer's Role

There are three kinds of  enthusiastic coding agent users:


1. Frustrated individual contributors who always wanted to be team managers so that they give orders and others execute but were never promoted.  Now they have a rubber doll coding agent to give orders to and have meetings with where they demand and inquire and the coding agents comply and report..
2. Frustrated  team managers (ראשי צוות) tired of stupid, non-compliant and arrogant  programmers who don't do their work right no matter what instructions are given. Now they have a wooden dummy coding agent to control, still stupid and non-compliant but at least humble and polite..
3. Chief surgeons (see attached) who can now replace some or all of their surgical team's members with iron men coding agents. They continue to perform the kind of work they always did, but are now less constrained by sickness leaves, bad moods,  or substance overuse..

Of these three kinds of enthusiastic coding agents users, in my humble opinion, only the third kind works. That means that the engineer's workflow remains intact, and what changes is how interactions with team members (now agents) are realized. In particular, what stays is that


• the engineer writes critical algorithmic code  in a programming language, because algorithms explained in 'plain English' never do what they seem to do, and there is no way to verify;.
• the engineer defines the success criteria explicitly via various tests that can be run automatically  (enforcement) and re-run reproducibly (regression); these can be done through contracts, success criteria, unit tests, functional tests, fuzzy tests etc. .
• the engineer specifies and maintains the project management tool setup (dependency management, project build, linting, unit testing, documentation builds, monitoring, dashboards etc.).
• everything that is done on the project aside from creative writing (debugging/testing/smoke tests/demonstration runs/....) is fully reproducible from managed artifacts..

What changes is that the coding agent can automate and ease these steps:


• the engineer can now write critical code in the language of their choice and care less about idiosyncratic optimizations; as an extreme example, the engineer can write the core algorithm in Haskell because Haskell is the best language for expressing that particular algorithm, even though the rest of the code is written in Python out of deployment consideration (even though Python is unsuitable for algorithmic coding);.
• success criteria has to be set forth so that the coding agent is forced to use them and cannot escape/report success unless formal criteria are met;.
• tools are now documented in AGENTS.md, skills, etc. rather than (just) in human-readable documentation, in a way  that lets the agents use them efficiently;.
• reproducibility is now testable rather than just planned --- with a human team checking that the same action routine can be performed again with the same or close results is expensive and demoralizing; with coding agents, this can be done as flow testing without physical, mental, or moral burden..

When you say "we do everything with claude" and mean that you prompt claude to death until it fails and then you hack what failed in an interactive debugger of your favorite IDE, you admit that you reproduce the worst traits of IDE development workflow using claude. It bears all the drawbacks of the old IDE use (no big picture, reliance on dumb code completion, limited and deceiving contextual help, non-reproducible work flows, opaque project management, environment lock-in, and more) and adds to that the slop that follows from the LLM's confirmation bias. This does not produce any good results. This seems to produce them and nurtures addiction to IDE+claude, which is detrimental.

Instead:

• be the chief surgeon;.
• take the responsibility for the tool set for all stages of your project that both you and the agent can use;.
• set forth success criteria explicitly, independently of what the LLM is trained to consider success (it is trained to maximize token burn, nothing else)..
