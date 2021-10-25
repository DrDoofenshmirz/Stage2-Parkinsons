
**Parkinson’s disease** is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages to it and affects more than 1 million individuals every year in India. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain.

**Parkinsons is characterised  by various symptoms** .
<p align="center">
<img src="https://user-images.githubusercontent.com/68779543/138735867-52693f6d-d427-4ce9-8b9a-fc7589de27ea.png" width="799" heigth="805">


The link for the  Dataset  used is given below 
* [Parkinsons Disease Dataset ](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/)

  # 📌Goal
The main goal of this project is to create a Machine Learning Classification Model that can classify a patient's case as Parkinsons or not depending on the various feature inputs we use to train the model.
  
 Abstract : Oxford Parkinson's Disease Detection Dataset
🚀 **WHATS IN THE DATASET?**

  Data Set Characteristics: Multivariate

  Number of Instances: 197

  Area: Life

  Attribute Characteristics: Real
  

  Number of Attributes: 23

  Date Donated: 2008-06-26

  Associated Tasks: Classification
  
  <h1 align="center"> Data Set Information</h1>
  
  <h2 align="center"> This dataset is composed of a range of biomedical voice measurements from 
31 people, 23 with Parkinson's disease (PD). Each column in the table is a 
particular voice measure, and each row corresponds one of 195 voice 
recording from these individuals ("name" column). The main aim of the data 
is to discriminate healthy people from those with PD, according to "status" 
column which is set to 0 for healthy and 1 for PD.

The data is in ASCII CSV format. The rows of the CSV file contain an 
instance corresponding to one voice recording. There are around six 
recordings per patient, the name of the patient is identified in the first 
column</h2>

👩‍🔬👩‍💻Workflow :
We will use the Google Colab Notebooks whih are popular data science tools to do our Model building.

1 Load the Dataset

2 Data Preprocessing

3 Splitting the dataset 

4 Training the model 

5 Testing The model

6 Determining accuracy

* Here we load all the libraries we will need to work with .

  ```
  import pandas as pandas
  from sklearn.preprocessing import MinMaxScaler
  from xgboost import XGBClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score```
 
 This is how some of the variables in the datasets values looks like.
 
 <img src="https://user-images.githubusercontent.com/68779543/138739239-fb7fc66b-9058-4638-9a65-7740da04d004.jpeg" width="700" heigth="900">
 
 We will first split the dataset such that X  will only have those columns that are going to be used for training while Y has the output .
 We can clearly see that the range of values vastly differ as some are in  the range of 1-10 while others are in hundreds & hence we shall first normalize the values using the MinMaxScaler function.
``` 
X = data.drop(['status' , 'name'] , axis = 'columns')
Y = data[['status']]
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(X)
Data= pd.DataFrame(x)
Data.head() 
```

 <img src="https://user-images.githubusercontent.com/68779543/138741025-714d56b1-2ebe-493d-9ab0-8a8279f036a1.jpeg" width="900" heigth="1200">

We have obtained the normalized values of the features in the dataset and thus  obtained more suitable values for training our model .

Now we have our dataset that we will split into training and testing dataset using the train_test_split() function.
```
x_train,x_test , y_train , y_test = train_test_split(x , Y  , test_size = 0.2)
```
<h1 align="center">🔬🖥️🔬🖥️ Training the model🔬🖥️🔬🖥️ </h1>
<h3 align="center">  we will be using the XGBoost which is a new Machine Learning algorithm designed with speed and performance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. In this project, we will import the XGBClassifier from the xgboost library; this is an implementation of the scikit-learn API for XGBoost classification </h3>

<img src="https://user-images.githubusercontent.com/68779543/138742101-6ee6d3c7-dd44-4879-92b0-feb247a352b4.jpg" width="900" heigth="500">

Execute the following code.
You can see what each code does  in the comments adjacent to it .

```
model = XGBClassifier() # calling the classifier
model.fit(x_train , y_train) #fitting the model ( training the dataset)
predictions = model.predict(x_test) # Making the trained model predict parkinsons predictions from test dataset 
```

The variable **predictions** contains all the prediction outputs for all the test dataset . 
We will determine the accuracy of our model by checking how many of our prediction are correct with respect to the  true status in the test dataset.
We do this by executing the following command.
```
accuracy_score(predictions , y_test) 
```
And the result was 94.87 % .

An accuracy value of 94.87 % is suitable enough for Machine learning  Model  for learning process.
 
The link for the Google colab notebook in Github is given below .
[Parkinsons Disease Prediction Notebook ] (https://github.com/almightxxx/Stage2-Parkinsons/blob/main/Detecting_Parkinsons.ipynb)

**The classification report**
![image](https://user-images.githubusercontent.com/68779543/138744204-6ab21488-3712-4760-a585-82e5e0cdfbaf.png)


