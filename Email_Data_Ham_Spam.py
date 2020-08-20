##############################################################################
########################## Naive-Bayes #######################################
##############################################################################
#Build a naive Bayes model on the data set for classifying the ham and spam

#importing packages and loading the data
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

Email_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Naive Bayes\\sms_raw_NB.csv",encoding = "ISO-8859-1")



###########################################################
################ DATA CLEANING ############################
###########################################################

import re
stop_words = []#creating a empty list

#reading a text file
with open("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Naive Bayes\\stop.txt") as f:
    stop_words = f.read()
   
    
##As Stopwards are in a single string, lets convert into list of single words
# splitting the entire string by giving separator as "\n" to get list of 
# all stop words 
stop_words = stop_words.split("\n")




#"this is awsome 1231312 $#%$# a i he yu nwj"
# ['This', 'is', 'Awsome', '1231312', '$#%$#', 'a', 'i', 'he', 'yu', 'nwj']
#def cleaning_text(i):
#    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
#    i = re.sub("[0-9" "]+"," ",i)
#    w = []
 #   for word in i.split(" "):
#        if len(word)>3:
#            w.append(word)
#    return (" ".join(w))#

#"This is Awsome 1231312 $#%$# a i he yu nwj".split(" ")#

#cleaning_text("This is Awsome 1231312 $#%$# a i he yu nwj")
#'this awsome'







##Defining a custom function for cleaning the data
def cleaningdata (i):
    i= re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w= []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))
    



##Applying the custome function to email data text column
#removes punctuations, numbers
Email_Data["text"]= Email_Data["text"].apply(cleaningdata)

#eg
#cleaningdata("Hope you are having a good week. Just checking in")
#cleaningdata("hope i can understand your feelings 123121. 123 hi how .. are you?")



##Removing the empty rows 
Email_Data.shape#(5559, 2)

Email_Data.isnull()
Email_Data = Email_Data.loc[Email_Data.text != " ",:]
##There are no empty spaces





# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation

# creating a matrix of token counts for the entire text document 
def split_if_words(i):
    return [word for word in i.split(" ")]

########## Feature Set(i/p Variables) ##############
predictors = Email_Data.iloc[:,1]
########### o/p Variable ###########
target = Email_Data.iloc[:,0]




######################################################################
####### Splitting the data into TRAIN and TEST data set ##############
######################################################################

from sklearn.model_selection import train_test_split
email_train,email_test = train_test_split(Email_Data,test_size=0.3)
x_train,x_test,y_train,y_test = train_test_split(predictors, target, test_size = 0.3, stratify = target)





# Preparing email texts into word count matrix format i.e bag of words
email_bow = CountVectorizer(analyzer = split_if_words).fit(Email_Data["text"])




#For all the mails doing the transformation
all_emails_matrix = email_bow.transform(Email_Data["text"])
all_emails_matrix.shape
#(5559, 6661)

#For training dataset(ie training emails)
train_emails_matrix = email_bow.transform(x_train)
train_emails_matrix.shape
#(3891, 6661)

##For test dataset( test mails)
test_emails_matrix = email_bow.transform(x_test)
test_emails_matrix.shape
##(1668, 6661)


#############################################################
############## MODEL BUILDING ###############################
#############################################################

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


########  Building the Multinomial naive bayes model #########

classifier_nb = MB()
classifier_nb.fit(train_emails_matrix,y_train)
train_pred_nb =classifier_nb.predict(train_emails_matrix) 
accuracy_nb = np.mean(train_pred_nb==y_train)#0.9879208429709586
#98.8%
pd.crosstab(train_pred_nb, y_train)
#type    ham  spam
#row_0            
#ham    3344    23
#spam     24   500

##predicting on test data
test_pred_nb = classifier_nb.predict(test_emails_matrix)
accuracy_test_nb = np.mean(test_pred_nb == y_test )
##96.82%
pd.crosstab(test_pred_nb,y_test)





############## Building Gaussian model #####################

classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),y_train.values)
train_pred_gb = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_gb = np.mean(train_pred_gb == y_train)#0.9108198406579285
#91.5%
pd.crosstab(train_pred_gb,y_train)
#type    ham  spam
#row_0            
#ham    3021     0
#spam    347   523

#predicting on test data
test_pred_gb = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_gb = np.mean(test_pred_gb == y_test)#0.842326139088729
#84.2%
pd.crosstab(test_pred_gb, y_test)
#type    ham  spam
#row_0            
#ham    1209    28
#spam    235   196



#########   Building with TFIDF transformation  #####################
# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape
#(3891, 6661)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape
#(1668, 6661)




#######   Building Multinomial Naive Bayes model ##################

classifer_mb_tfidf = MB()
classifer_mb_tfidf.fit(train_tfidf,y_train)
train_predmb_tfidf = classifer_mb_tfidf.predict(train_tfidf)
accuracy_mb_tfidf = np.mean(train_predmb_tfidf == y_train)#0.9668465690053971
#96.7%
pd.crosstab(train_predmb_tfidf, y_train)
#type    ham  spam
#row_0            
#ham    3368   129
#spam      0   394
test_predmb_tfidf = classifer_mb_tfidf.predict(test_tfidf)
accuracy_testmb_tfidf = np.mean(test_predmb_tfidf == y_test)#0.9538369304556354
#95.3%



#######  Building gaussiam naive bayes model  #####################

classifier_gb_tfidf = GB()
classifier_gb_tfidf.fit(train_tfidf.toarray(),y_train.values)
train_predgb_tfidf = classifier_gb_tfidf.predict(train_tfidf.toarray())
accuracy_gb_tfidf = np.mean(train_predgb_tfidf == y_train)#0.9108198406579285
#91%
pd.crosstab(train_predgb_tfidf,y_train)
#type    ham  spam
#row_0            
#ham    3021     0
#spam    347   523
test_predgb_tfidf = classifier_gb_tfidf.predict(test_tfidf.toarray())
accuracy_testgb_tfidf = np.mean(test_predgb_tfidf == y_test)#0.8369304556354916
#83.7%
pd.crosstab(test_predgb_tfidf,y_test)
#type    ham  spam
#row_0            
#ham    1211    39
#spam    233   185
# inplace of tfidf we can also use train_emails_matrix and test_emails_matrix instead of term inverse document frequency matrix 



#the Multinomial naive bayes model have the highest accuracy