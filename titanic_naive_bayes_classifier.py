import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
titanic_df=pd.read_csv("data/Titanic-Dataset.csv")
#prof = ProfileReport(titanic_df)
#prof.to_file(output_file='titanic_output.html')
titanic_df.drop(['PassengerId', 'Name','Cabin','Ticket','SibSp'], axis='columns', inplace=True)
#sns.pairplot(titanic_df, hue='Survived')
#plt.show()
#create a pipeline for handling numerical data
numerical_features = ['Pclass','Age','Parch','Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
#create another pipeline for handling categorical data
categorical_features = ['Sex','Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',OneHotEncoder(handle_unknown='ignore',drop='first'))
])
#combine the two pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#split the data in train and test
X_train,X_test,y_train,y_test = train_test_split(titanic_df.drop(columns=['Survived']),
                                                 titanic_df['Survived'],test_size=0.2)
GNB_model = GaussianNB()
GNB_classifier=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GNB_model)
])
GNB_classifier.fit(X_train,y_train)
print("GNB model Score:",GNB_classifier.score(X_test,y_test))
y_prediction=GNB_classifier.predict(X_test)
#Generate report for classifier
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
