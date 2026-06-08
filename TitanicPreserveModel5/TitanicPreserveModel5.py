import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#--------------------------------------------------------------
# function name : TrainTitanicModel
# description : it does split X, Y training data, testing data
# parameters : display info
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------
def TrainTitanicModel(df):
    # split featrures and labels
    X = df.drop("Survived",axsis = 1 )
    Y = df["Survived"]

    print("\nFeatures")
    print(X.head())

    print("\nLables : ")
    print(Y.head())

    print("shape of X : ",X.shape)
    print("shape of Y : ",Y.shape)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)

    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train,Y_train)
    print("model train successfully")

    print("\nIntercept of model : ")
    print(model.intercept_)

    print("\nCoefficient of model")
    for feature , coeficient in zip(X.columns, model.coef_[0]):
        print(feature, ":",coeficient)

#--------------------------------------------------------------
# function name : PreservedModel
# description : it is used to preserve model on secondary
# parameters : display info
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------
def  PreservedModel(model,filename):
    joblib.dump(model,filename)

    print("Model preserved sucessfully with name :", PreservedModel)
    PreservedModel(model,"Marvelloustitanic.pkl")

#--------------------------------------------------------------
# function name : MarvellousTitaniLogistic
# description : this is main pipeline controller
# parameters : display info
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------

def DisplayInfo(title):
    print("\n" + "="*70)
    print(title)
    print("-"*70)

#--------------------------------------------------------------
# function name : show data
# description : it shows basic information about dataset
# parameters : dataset (df)
#               df --> pandas dataframe object
#              message
#              message --> Heading text to display
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------

def ShowData(df,message):
    DisplayInfo(message)

    print("First 5 file rows of dataset ")
    print(df.head())

    print("\nShape of datset")
    print(df.shape)

    print("\nColumn names : ")
    print(df.columns.tolist())

    print("\nMissing values in each column")
    print(df.isnull().sum())


#--------------------------------------------------------------
# function name : CleanTitaniData
# description :  it does preprocessing
#                it removed unnecessary columns
#                it handles missing values
#                it converts text data to numeric format
#                it does encoding to categorical columns
#            
# parameters : df --> pandas dataframe
# return :  df --> clean pandas dataframe
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original data")
    print(df.head())

    # removed unecessary coloumns
    drop_columns = ["Passangerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)

    # drop the unwanted columns
    df = df.drop(columns = existing_columns)
    DisplayInfo("Step 2 : Original data after columns removal")
    print(df.head())

    # handle age column
    if "Age" in df.columns:
        print("Age columns before filling missing values")
        print(df["Age"].head(10))

        # invalid value gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        age_median = df["Age"].median()
        
        # replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing: ")
        print(df["Age"].head(10))

    # handle fare column
    if "Fare" in df.colulmns:
        print("\n Fare column before preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        fear_median = df["Fare"].median()

        

    # Handle embarked column
    if "Embarked" in df.colulmns:
        print("\n Embarked column before preprocessing")
        print(df["Embarked"].head(10))

        df["Embarked"]= df["Embarked"].astype(str).str.strip()

    # remove missing values
        df["Embarked"] = df["Embarked"].reaplace(['nan','None',''],np.nan) 
     
    # get most frequent vlaue
        embarked_mode = df["Embarked"].mode()[0]

        print("mode of embarked column : ",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\Embarked column before preprocessing")
        print(df["Embarked"].head(10))

    if "sex" in df.colulmns:
        print("\n sex column before preprocessing")
        print(df["sex"].head(10))

        df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
        fear_median = df["sex"].median()
        print("\nsex column after preprocessing")
        print(df["sex"].head(10))

    DisplayInfo("Data after preprocessing")
    print(df.head())

    print("\nMissing values after preprocessing")
    print(df.isnull().sum())


    # encode Embarked column
    
    df = pd.get_dummies(df, columns=["Embarked"],drop_first=True)
    print("\n data after encoding")

    print(df.head())
    print("Shape of dataset ",df.shape)

    # convert boolean columns into integer

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)       


     

    return df


#--------------------------------------------------------------
# function name :TitaniLogistic
# description : this is main pipeline controller
#               it loads the dataset, show raw data
#               it preprocess the dataset nd train the model 
# parameters : data path of dataset file
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------


def TitaniLogistic(Datapath):
    DisplayInfo("Step : Loading the dataset")
    df = pd.read_csv(Datapath)

    ShowData(df,"initial dataset")

    TrainTitanicModel(df)








#--------------------------------------------------------------
# function name : main
# description : starting point of the application
# parameters : none
# return : none
# data : 14/3/2026
# author : shruti Dabhade 
#--------------------------------------------------------------


def main():
    TitaniLogistic("MarvellousTitanicDataset.csv")






if __name__=="__main__":
    main()