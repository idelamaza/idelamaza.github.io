---
layout: single
title: "Data Science Bowl 2019"
date: 2019-12-05
categories: [Data Science]
tags: [Kaggle]
toc: true
excerpt: "Predicting children's performance in educational game apps"
author_profile: true
header:
  image: "/images/posts/2019-12-05-Data-Science-Bowl-2019/header.jpg"
  caption: "Photo by [Kelly Sikkema](https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/)"
---

This work was presented as the course final project for MIT 15.095: _Machine Learning Under a Modern Optimization Lens_, by Prof. Dimitris Bertsimas. A printer-friendly version of the whole report can be downloaded [here]({{ site.url }}{{ site.baseurl }}/files/2019-12-05-Data-Science-Bowl-2019/report.pdf), and a summary in the form of a poster can be found [here]({{ site.url }}{{ site.baseurl }}/files/2019-12-05-Data-Science-Bowl-2019/poster.pdf). All the used code can be found at [GitHub](https://github.com/inigodelamaza/Data-Science-Bowl-2019). Special thanks to my colleague and co-author of the project, [Juan José Garau](http://systemarchitect.mit.edu/students.php#garau).

The project comprises the study of the application of predictive, optimization-based classification algorithms to a real-world task. Looking towards that aim, an online data science competition is the perfect medium due to both the availability of high-quality data, and the possibility of comparing different models and outcomes with the ones achieved by other participants using conventional heuristic methods. The online platform Kaggle is one of the most popular online competition organizer and 2019's edition of the [Data Science Bowl®](https://www.kaggle.com/c/data-science-bowl-2019/overview "Data Science Bowl®")(DSB) is a perfect fit for the objectives of this project.

## Introduction

Online apps are a key element in the democratization of education all over the world, specially when it comes to early childhood education. Developing software focused to improve how kids learn is essential to make an impact on the users’ abilities and skills. In this year's edition of the DSB, company PBS KIDS, a trusted name in early childhood education (3-5 years old) for decades, aims to gain insights into how media and game mobile apps can help children learn important skills for success in school and life. In this work, an optimization-based approach to understand the relationship between the educational content of the PBS KIDS Measure UP! app and how effectively its users learn is presented.

### Problem statement

The app has a diverse range of educational material including videos, assessments, activities, and games. These are designed to help kids learn different measurement concepts such as weight, capacity, or length. The DSB challenge focuses on predicting players’ assessment performance based on the information the app gathers on each game session. Every detailed interaction a player has with the app is recorded in the form of an event, including things like mediacontent display, touchscreen coordinates, and assessment answers. 

This data is highly fine-grained, as multiple events might be recorded in less than one second. With this data, the competition participants must predict the performance ofcertain assessments whose outcomes are unknown, given the history of a player up to that assessment. Based on the prediction, the goal is to __classify__ the player into one of four performance groups, which represent how many attempts takes a kid to successfully complete the assessment. The aim of the challenge is to help PBS KIDS discover important relationships between engagement with high-quality educational media and learning processes, by understanding what influences the most on the childrens’ performance and learning rate.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/pbs-app.png" alt="">
  <figcaption>A look at PBS KIDS Measure Up! app.</figcaption>
</figure> 

We tackle the problem using a feature extraction process focused on exploiting the time series nature of the data and the application of different Optimal Classification Tree ([Bertsimas & Dunn, 2019](#bertsimas2019)) models in order to both achieve a high predictive power and gain interpretable insights. Additionally, we provide two ideas of how this work could continue leveraging the potential of optimization-based approaches.

## Data Overview

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/data_overview.png" alt="">
  <figcaption>Dataset structure.</figcaption>
</figure> 

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/label_dist.png" alt="">
  <figcaption>Prediction label distribution.</figcaption>
</figure> 

## Feature Engineering

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/feat_eng.png" alt="">
  <figcaption>Instance definition approach.</figcaption>
</figure> 

<figure>
	<div style="position: relative; padding-bottom: 75%; height:0;overflow: hidden;"> 
	  <iframe width= "100%" height= "100%" frameborder="0" scrolling="yes" id="igraph" seamless="seamless" top="0" left="0" position="absolute" src="/charts/2019-12-05-Data-Science-Bowl-2019/timeseries2.html">
	  </iframe>
	</div>
  <figcaption>Time series representation of the game_sessions of a certain installation_id. The annotation above certain game_session shows the number of events they comprise (notice that all Clips (green) have only one event, as there is no interaction of the player recorded).</figcaption>
</figure>

## Optimal Classification Trees

<figure>
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="yes" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_no_autobalance.html">
  </iframe>
  <figcaption>OCT without the autobalance setting.</figcaption>
</figure>

<figure>
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="no" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_autobalance.html">
  </iframe>
  <figcaption>OCT with autobalance.</figcaption>
</figure>

## Additional optimization-based approaches



## Conclusions and Future Work



## References

<a name="bertsimas2019"></a>
Bertsimas, D., & Dunn, J. (2019). _Machine learning under a modern optimization lens._ Belmont, MA: Dynamic Ideas LLC. Available at: [https://www.dynamic-ideas.com/books/machine-learning-under-a-modern-optimization-lens](https://www.dynamic-ideas.com/books/machine-learning-under-a-modern-optimization-lens).