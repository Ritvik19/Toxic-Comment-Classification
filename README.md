# Toxic-Comment-Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

Some characteristics that can signify that a text is toxic:

* Has a non-neutral tone
  * Has an exaggerated tone to underscore a point about a group of people
  * Is rhetorical and meant to imply a statement about a group of people
* Is disparaging or inflammatory
  * Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
  * Makes disparaging attacks/insults against a specific person or group of people
  * Based on an outlandish premise about a group of people
  * Disparages against a characteristic that is not fixable and not measurable
* Isn't grounded in reality
  * Based on false information, or contains absurd assumptions
* Uses sexual content (incest, bestiality, pedophilia) for shock value

**Problem Statement:** to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate

**Sources:** [Kaggle-Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) and [Kaggle-Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/)

**Project Objective:** a model to prerform advanced sentiment analysis

___

### Approach Summary

**Performance Measure:** Area Under Receiver Operating Characteristic

**Feature Extraction:** Sublinear Smoothed TFIDF

**Algorithm:** Multinomial Naive Bayes and Logistic Regression

___

### Performance Summary

Selected | Approach | Algorithm | Mean AUROC | Mean Accuracy | Mean F1
:---:|:---|:---|---:|---:|---:
&#9744; | One Vs Rest | Naive Bayes | 0.8176 | 0.9335 | 0.0835
&#9744; | One Vs Rest | Logistic Regression | 0.9765 | 0.8576 | 0.5277
&#9745; | Random Oversampling | Naive Bayes | 0.9193 | 0.8832 | 0.9184
&#9745; | Random Oversampling | Logistic Regression | 0.9738 | 0.9551 | 0.9441
&#9745; | SMOTE | Naive Bayes | 0.9185 | 0.9163 | 0.9391
&#9745; | SMOTE | Logistic Regression | 0.9718 | 0.9651 | 0.9691