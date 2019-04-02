---
title: "A Go Transgression"
date: 2019-04-02T10:19:13+03:00
draft: False
---

[Go](http://golang.org/) gives the programmer introspection into
every aspect of [the language](https://godoc.org/reflect), and
of a [running program](https://godoc.org/runtime). But to one
thing the programmer does not have access, and it is the
goroutine identifier. Because the day the programmers know the
goroutine identifier, they create goroutine-local storage
through shared access and mutexes, and [shall surely
die](https://www.kingjamesbibleonline.org/Genesis-2-17/). 

In [Infergo](https://infergo.org/), I had to have
goroutine-local storage. Here is how I got [efficient
goroutine-local storage in Go](http://infergo.org/news/tale-of-goids).
