# Pneumonia-Project-using-Svm
Problem Statement:  
Pneumonia is a serious infection lung infection, and early detection is crucial for treatment.
Chest X-rays are commonly used for diagnosis, but manual interpretation can be time-consuming and error Prone.
Build a ML model to classify chest X-ray as images as Pneumonia or Normal using Pneumoniamnist data  

Approach
Applied Support Vector Machine (SVM) for classification.
Implemented data augmentation to enhance model robustness.
SVM was chosen for its effectiveness in high dimensional spaces and its robustness to overfitting, especially in scenarios with limited data.

Data Preprocessing  
Dataset Loading:      The Pneumoniamnist.npz files was loaded into the jupyter.
Feature Flattening :  This structure is commonly used in machine learning tasks, where the training set is used to train models, and the testing set is used to evaluate their performance

Classification: 
Model choice: Trained a Support Vector Machine classifier on the extracted features.
Hyperparameter Tuning: Employed grid search with cross-validation to optimize the SVM's regularization parameter (C) and kernel parameters. 

Model Evaluation:
Cross-Validation: Implemented k-fold cross-validation to assess model performance and ensure generalizability.

Model Modifications:
Class Weight Adjustment: Set class weights inversely proportional to class frequencies to address class imbalance.
Feature Scaling: Applied StandardScaler to normalize feature vectors before SVM training.
Confusion Matrix
True Positives (TP):    688 (Pneumonia correctly predicted as Pneumonia)
True Negatives (TN):  59 (Normal correctly predicted as Normal)
False Positives (FP):   19 (Normal incorrectly predicted as Pneumonia)
False Negatives (FN): 11 (Pneumonia incorrectly predicted as Normal


Precision: The 96% of the total prediction positive predictions are actually correct.
Recall (Sensitivity): The 97% actual Pneumonia positives that are correctly identified.
Accuracy: Provides a general measure of the model's correctness. However, it can be misleading in imbalanced datasets.

Mitigate Class Imbalance 
Support Vector Classifier (SVC), the class_weight parameter allows to assign different weights to different classes. This is particularly useful when dealing with imbalanced datasets, where one class has significantly more samples than the other. By assigning higher weights to the minority class, the model is penalized more for misclassifying instances of that class, encouraging it to pay more attention to those instances.


Measures taken to prevent over-fitting 
Cross-validation is a robust technique for evaluating machine learning models. 
Assessing Model Performance: By training and testing the model on different subsets of the data, it provides a more reliable estimate of its performance.
Detecting Overfitting: If a model performs well on the training data but poorly on the validation data, it may be overfitting. Cross-validation helps in identifying such issues.
Hyperparameter Tuning: It allows for tuning model parameters (like C and gamma in SVC) by evaluating their performance across different folds.




