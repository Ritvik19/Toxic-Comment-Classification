# Toxic-Comment-Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

**Problem Statement:** to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate

**Sources:** [Kaggle-Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) and [Kaggle-Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/)

**Project Objective:** a model to prerform advanced sentiment analysis

___

### Approach Summary

**Performance Measure:** Area Under Receiver Operating Characteristic

**Feature Extraction:** Sublinear Smoothed TFIDF

**Algorithm:** Logistic Regression

___

### Performance Summary

Selected | Approach | Mean AUROC | Mean Accuracy
:---:|:---|---:|---:
&#9744; | One Vs Rest | 0.973 | 78.6%
&#9744; | NB Featurer | 0.9748 | 94.92%
&#9744; | Over Sampling | 0.9826 | 94.90%
&#9745; | SMOTE | 0.9875 | 95.61%