#We'll apply different naive bayes algorithm on different data set
#titanic_naive_bayes_classifier.py
1. EDA
We generated the profile report (see titanic_output.html) of the dat set using ProfileReport  from ydata_profiling package.
We observed, Age column has 177 records missing and Cabin column has 687 records missing. 
We dropped 'PassengerId', 'Name' and 'Cabin' columns and drew pair plot with rest of the features.
We decided to drop 'PassengerId', 'Name','Cabin','Ticket' and 'SibSp' features based on EDA.
2. Data preprocessing
Handling missing values
The Age column has 177 and Embarked has 2 records missing. We used mean of age and most frequent values for Embarked column to impute the missing values.
Categorical encoding
We applied OHE on 'Sex' and 'Embarked' column to change them to numeric values
Normalize the data
Then, we applied standard scaler to all the data to normalize them.
3. We applied Gaussian Naive Bayes classifier on the data and the score is
GNB model Score: 0.8044692737430168
Accuracy: 0.8044692737430168
Confusion matrix     0   1
0  99  14
1  21  45
Precision:   0.8020334248650695
Recall:  0.8044692737430168
F1 score:  0.8019315702400076