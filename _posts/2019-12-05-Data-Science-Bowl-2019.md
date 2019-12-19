---
layout: single
title: "Data Science Bowl 2019"
date: 2019-12-05
categories: [Projects]
tags: [Data Science, Optimization]
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
  <figcaption>Figure 1. A look at PBS KIDS Measure Up! app.</figcaption>
</figure> 

We tackle the problem using a feature extraction process focused on exploiting the time series nature of the data and the application of different Optimal Classification Tree ([Bertsimas & Dunn, 2019](#bertsimas2019)) models in order to both achieve a high predictive power and gain interpretable insights. Additionally, we provide two ideas of how this work could continue leveraging the potential of optimization-based approaches.

## Data Overview

The dataset provided by the competition is primarily composed of all the game analytics the PBS KIDS Measure UP! app gathers anonymously from its users. In this app, children navigate through a map and complete various levels, which may be activities, video clips, games, or assessments (each of them considered a different ``game_session``, and its nature expressed as ``type``).  

Each assessment is designed to test a child’s comprehension of a certain set of measurement-related skills. There are five assessments: _Bird Measurer_, _Cart Balancer_, _Cauldron Filler_, _Chest Sorter_, and _Mushroom Sorter_. 

### Data volume

To help the reader understand the data, Figure 2 shows the tree-like structure of the dataset. Each application install is represented by an ``installation_id``, which can be considered to map to one single user. The training set provides the full history of gameplay data of 17,000 ``installation_ids``, while the test set has information for 1,000 players.  Moreover, each ``installation_id`` has multiple ``game_sessions`` of different types (activities, games, assessments...). In total there are around 300,000 ``game_sessions`` in the training set and almost 30,000 in thetest set. 

Finally, each ``game_session`` is composed by several events that represent every possible interaction between the user and the app. These events are identified with a unique ID (``event_id``), and have associated data such as screen coordinates, timestamps, durations, etc, depending on the nature of the event. In total, there are around 11.3M events in the training set and 1.1M in the test set. We can conclude the dimensionality of the problem is high and the data is presented in the form of dependencies and __time series__.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/data_overview.png" alt="">
  <figcaption>Figure 2. Dataset structure.</figcaption>
</figure> 

### Classification labels

As previously introduced, the output of the model must be a prediction of the performance of the children in a particular assessment, in the form of a classification. The model should select one performance group out of four possible ``accuracy_groups``. The groups are numbered from 0 to 3 and are defined as follows:

- ``accuracy_group3`` : the assessment was solved on the first attempt
- ``accuracy_group2`` : the assessment was solved on the second attempt
- ``accuracy_group1`` : the assessment was solved after three or more attempts
- ``accuracy_group0`` : the assessment was was never solved

Figure 3 shows the distribution of the four possible ``accuracy_groups`` across the five assessments of the app. Notice that there are clearly three assessments that seem to be easier _a priori_ as there is a majority of first-attempt correct answers, whereas one them has pretty much evenly distributed labels (_Bird Measurer_) and another one (_Chest Sorter_) seems to be more difficult than the others.

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/label_dist.png" alt="">
  <figcaption>Figure 3. Prediction label distribution.</figcaption>
</figure> 

### Training and test sets differences

Both training and test sets provided by the competition have similar structure in terms of the amount of information provided for each recorded event. The main difference is that all the information for each user (``installation_id``) in the test set stops at the beginning of an assessment, whose outcome is meant to be predicted. 

However, in many cases, information from previous complete assessments is given. Every ``installation_id`` in the test has at least one assessment (the one to be labeled by the model), whereas many ``installation_id`` from the training set do not have any, and therefore have been removed from the dataset because they do not allow to generate any label for the training process. This leads to a major data volume decrease, as only 4242 ``installation_id`` out of the 17,000 original ones can be used.

## Feature Engineering

The process of extracting features to train the machine learning model is time-consuming and challenging in this specific competition. Not only the provided data is in the form of a time series, but it also has a large amount of unnecessary information that needs to be filtered. In this section the process of transitioning from the original dataset to a dataset suitable to make predictions upon is explained.

### Structuring data as time series

<figure>
  <iframe width= "800" height= "500" frameborder= "0" scrolling="no" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/timeseries.html">
  </iframe>
  <figcaption>Figure 4. (interactive) Time series representation of the game_sessions of a certain installation_id. The annotation above certain game_session shows the number of events they comprise (notice that all Clips (green) have only one event, as there is no interaction of the player recorded).</figcaption>
</figure>

### Training instance definition

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/feat_eng.png" alt="">
  <figcaption>Figure 5. Instance definition approach.</figcaption>
</figure> 

### Feature definition

#### Assessment-related features

#### Player-related features

## Optimal Classification Trees

Since the task is to predict the accuracy group of a player before taking an assessment, in this work we focus onoptimization-based classification models, namely Optimal Classification Trees (OCT). By means of Julia’s packageIAI, in this section we train several OCT models and analyze their performance. All the code that allows us to obtainthe performance results from this section is presented in the second and third notebooks appended at the end of thisreport.

### Models

Understanding the tradeoff between interpretability and predictive power they have, in this work we consider OCTmodels with both parallel and hyperplane splits in order to gain insights on what makes users perform well and to obtainmodels capable of reaching high scores in the competition, respectively.5
We understand the importance of training models with an appropriate selection of hyperparameters, therefore we carryout hyperparameter tuning tasks with a validation set before evaluating the performance of each model. In the case ofOCT with parallel splits, we create a grid that considers three different values for themaximum depthparameter (5, 10,and 15) and three different values for theminbucketparameter (1, 5, and 10). In the case of OCT-H, in order not tocompromise runtime given the size of the dataset, we choose to fix theminbucketparameter to 1 and train two modelswith amaximum depthof 2 and 3, respectively. Since OCT-H considers hyperplane splits, and we decide not to enforceany sparsity constraint, choosing any other depth value beyond the selected ones does not provide impactful predictivepower, degrades the interpretability of these models, and unnecessarily increases runtime.All models are trained using theOptimalTreeClassifierwith parallelization and using the misclassification erroras criterion to determine the best combination of hyperparameters. Additionally, we decide to duplicate the numberof analyses and repeat both the OCT and OCT-H cases choosing to equally prioritize the 4 classes by means of theautobalanceattribute. The rationale behind this decision can be easily understood by looking at figures 5 and 6 in theAppendix. We cover the details of these figures in the coming section but the reader can notice that whenautobalanceis not used there is no leaf that predicts class 2. This is corrected when enforcing an equal importance among classes,helping understand the relationship between players’ behaviour and this class.

### Results - Predictive power

Tables 1 and 2 show the training and out-of-sample performance of the OCT and OCT-H models with and without theautobalanceattribute. In this work we analyze the performance using two different metrics: on one hand we considerthe accuracy, which takes into account the proportion of correctly classified labels; on the other hand we use the Cohen’skappa coefficient with quadratic weights, which is the official metric of the competition and is defined as \\[\kappa = 1 - \frac{\sum_{i=1}^k\sum_{j=1}^kw_{ij}x_{ij}}{\sum_{i=1}^k\sum_{j=1}^kw_{ij}m_{ij}}\\] where \\(x_{ij}\\) represents the number of datapoints from class \\(i\\) that have been classified as class \\(j\\), \\(m_{ij}\\) is the expected number of datapoints from class \\(i\\) that have been classified as class \\(j\\), and finally \\(w_{ij}\\) is a weight factor defined as \\[w_{ij} = 1 - \frac{i^2}{(j-i)^2}\\]

By looking at the tables we can immediately appreciate the dominance of OCT-H models regardless of the use of the _autobalance_ attribute. The _autobalance_ feature works as expected, models without it perform better but that entails completely missing a class. This can be observed in figures [A1](#appendix) and [A2](#appendix) from the appendix, which do not have any leaf predicting class 2 (kids who solved the assessment in the second attempt).

<figure class="align-center">
  <img width = "100" src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/training_performance.png" alt="">
  <figcaption>Figure 6. Training performance of OCT and OCT-H with and without the autobalance attribute.</figcaption>
</figure> 

<figure class="align-center">
  <img width = "100" src="{{ site.url }}{{ site.baseurl }}/images/posts/2019-12-05-Data-Science-Bowl-2019/out_of_sample_performance.png" alt="">
  <figcaption>Figure 7. Out-of-sample performance of OCT and OCT-H with and without the autobalance attribute.</figcaption>
</figure> 

\\[ a^2 = b^2 \\] 

<figure>
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="yes" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_no_autobalance.html">
  </iframe>
  <figcaption>Figure 8. (interactive) OCT without the autobalance setting.</figcaption>
</figure>

<figure>
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="no" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_autobalance.html">
  </iframe>
  <figcaption>Figure 9. (interactive) OCT with autobalance.</figcaption>
</figure>

## Additional optimization-based approaches



## Conclusions and Future Work

## <a name="appendix"></a>Appendix

<figure style="width: 1200px">
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="yes" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_no_autobalance.html">
  </iframe>
  <figcaption>Figure A1. (interactive) OCT without the autobalance setting.</figcaption>
</figure>

<figure style="width: 1200px">
  <iframe width= "100%" height= "400" frameborder= "0" scrolling="no" id="igraph" seamless="seamless" src="/charts/2019-12-05-Data-Science-Bowl-2019/tree_oct_autobalance.html">
  </iframe>
  <figcaption>Figure A2. (interactive) OCT with autobalance.</figcaption>
</figure>

## References

<a name="bertsimas2019"></a>
Bertsimas, D., & Dunn, J. (2019). _Machine learning under a modern optimization lens._ Belmont, MA: Dynamic Ideas LLC. Available at: [https://www.dynamic-ideas.com/books/machine-learning-under-a-modern-optimization-lens](https://www.dynamic-ideas.com/books/machine-learning-under-a-modern-optimization-lens).