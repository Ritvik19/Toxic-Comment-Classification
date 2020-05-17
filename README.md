# Toxic-Comment-Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

**Problem Statement:** to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate

**Source:** [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)

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
&#9744; | One Vs Rest | 0.9765 | 85.06%
&#9744; | NB Featurer | 0.9786 | 96.28%
&#9744; | Over Sampling | 0.9857 | 95.32%
&#9745; | SMOTE | 0.9835 | 94.88%