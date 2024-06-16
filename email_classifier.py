import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
email_df=pd.read_csv("data/combined_data.csv")
print("data:",email_df.info())
print(email_df.groupby("label").describe())
print("Duplicate values:",	email_df.duplicated().sum())
# split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(email_df.text,
                                                    email_df.label,test_size=0.2)
classification_pipe = Pipeline(steps=[
    ('numeric_conversion', CountVectorizer()),
    ('classifier', MultinomialNB())
])
classification_pipe.fit(X_train,y_train)
print("Multinomial NB model Score:",classification_pipe.score(X_test,y_test))
y_prediction=classification_pipe.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score, confusion_matrix
# Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_prediction))
#confusion_matrix
confusion_matrix=pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,2)))
print("Confusion matrix",confusion_matrix.head())
print("Precision:  ",precision_score(y_test,y_prediction,average='weighted'))
print("Recall: ",recall_score(y_test,y_prediction,average='weighted'))
print("F1 score: ",f1_score(y_test,y_prediction,average='weighted'))
print("-"*80)
#another way is using classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_prediction))
