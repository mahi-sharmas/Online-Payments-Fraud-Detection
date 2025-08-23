![Project Screenshot](images/online-payments.png)
Online payment is the most popular transaction method in the world today. However, with an increase in online payments also comes a rise in payment fraud. The objective of this notebook is to train machine learning models for identifying fraudulent and non-fraudulent payments. The dataset is collected from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments.

About the Dataset:

We will be using the following columns in the dataset: There are 10 columns.

Step: tells about the unit of time
Type: type of transaction done
Amount: the total amount of transaction
nameOrg: account that starts the transaction
oldbalanceOrg: balance of the account of sender before transaction
newbalanceOrg: balance of the account of sender after transaction
nameDest: account that receives the transaction
oldbalanceDest: balance of the account of receiver before transaction
newbalanceDest: balance of the account of receiver after transaction
isFraud: the value to be predicted i.e. 0 or 1
Importing Libraries and Datasets

The libraries used are:

Pandas
Seaborn/Matplotlib
Numpy
The dataset includes features like type of payment, old balance, new balance, amount paid, name of destination, etc.
