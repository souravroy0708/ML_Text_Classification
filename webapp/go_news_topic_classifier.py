from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
#from textblob import TextBlob
import re
from collections import namedtuple
from os import listdir
from os.path import isfile, join
import multiprocessing
import datetime
import os

tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import pickle
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


from pymongo import MongoClient
client =  MongoClient('localhost',27017)
database = client.news

topic_map_zip = dict(zip(["sports","entertainment","political","finance","technology","health","education","business"],[1,2,3,4,5,6,7,8]))
topic_list = ["sports","entertainment","political","finance","technology","health","education","business"]

class TopicClassifier:
    def get_token(self,text):
        #Remove all special character
        clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
        tokens = tokenizer.tokenize(clean_text)
        tokens = [token.lower() for token in tokens
                      if token not in stopwords and len(token) > 3]
        return tokens

    def get_train_data(self,model_test_size=.001):
        records = database.doc2vec_news.find({"topic":{"$in":topic_list,"$exists":True}})

        X = []
        y = []
        counter =0
        for record in records:
            topic_text = record.get("topic")
            if not topic_text:
                continue;
            topic_int = topic_map_zip[topic_text]
            description = record.get("description")
            words = self.get_token(description)
            words_text = ' '.join(words)
            X.append(words_text)
            y.append(topic_int)
            counter +=1
        print ("record procced %s" %(counter))
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=model_test_size)
        print ("Traing set : %s Testing set : %s Total sample : %s" %(len(X_train),len(X_test),len(X)))
        return X_train, X_test, y_train, y_test

    def save_dtm_vector(self):
        X_train, X_test, y_train, y_test = self.get_train_data()
        vect = CountVectorizer(min_df=2,stop_words='english')
        # learn training data vocabulary, then use it to create a document-term matrix
        vect.fit(X_train)
        vect.transform(X_train)
        abs_path = os.getcwd()
        doc_vec_file = "%s/doc2vec_files/%s" %(abs_path,'go_news_dtm.pkl')
        pickle.dump(vect, open(doc_vec_file, 'wb'))
        print("Vector ---",vect)
        return doc_vec_file
    #save_dtm_vector()
    
    def train_clasification_model(self,model_name="nb"):
        # self.save_dtm_vector()
        abs_path = os.getcwd()
        doc_vec_file = "%s/doc2vec_files/%s" %(abs_path,'go_news_dtm.pkl')
        model_vector = pickle.load(open(doc_vec_file, 'rb'))
        if model_name == "nb":
            train_model = MultinomialNB()
            print ("NB model selected")
        elif model_name == "logreg":
            train_model = LogisticRegression()
            print ("Logreg model selected")
        X_train, X_test, y_train, y_test = self.get_train_data()
        model_vector.fit(X_train)
        X_train_dtm = model_vector.transform(X_train)
        X_test_dtm = model_vector.transform(X_test)
        # train the model using X_train_dtm (timing it with an IPython "magic command")

        train_model.fit(X_train_dtm, y_train)
        # make class predictions for X_test_dtm
        abs_path = os.getcwd()
        saved_model_name = "%s/doc2vec_files/%s_go_news_model.pkl" %(abs_path,model_name)
        save_model= pickle.dump(train_model, open(saved_model_name, 'wb'))
        y_pred_class = train_model.predict(X_test_dtm)
        print ("%s : y_pred_class -- %s original_class : %s " %(model_name,y_pred_class,y_test))
        # calculate accuracy of class predictions

        acc = metrics.accuracy_score(y_test, y_pred_class)
        print ("%s :  acc -- %s" %(model_name,acc))

        # calculate predicted probabilities for X_test_dtm (well calibrated)
        y_pred_prob = train_model.predict_proba(X_test_dtm)
        print("%s : y_pred_proba %s " %(model_name,y_pred_prob))
        return saved_model_name
    
    def get_topic_name(self,id=0):
        topic = ""
        if id:
            for topic_name,value in topic_map_zip.items():
                if value == id:
                    topic = topic_name
                    break
        return topic
   
    def get_prediction_topic(self,model_name="nb",document=""):
        abs_path = os.getcwd()
        saved_model_name = "%s/doc2vec_files/%s_go_news_model.pkl" %(abs_path,model_name)
        model = pickle.load(open(saved_model_name, 'rb'))
        doc_vec_file = "%s/doc2vec_files/%s" %(abs_path,'go_news_dtm.pkl')
        model_vector = pickle.load(open(doc_vec_file, 'rb'))
        document_matrix = model_vector.transform([document])
        y_predict = model.predict(document_matrix)
        try:
            y_pred_prob = model.predict_proba(document_matrix)[0]
            pred_details=[]
            counter = 0
            for pred_score in y_pred_prob:
                pred_detail={}
                pred_topic = topic_list[counter]  
                pred_detail["topic"] = pred_topic
                try:
                    score_percentage = pred_score*100
                    score_percentage =("%.2f" % round(score_percentage,2))
                except:
                    score_percentage = 0

                if score_percentage == "100.00":
                    score_percentage = 100
                if score_percentage == "0.00":
                    score_percentage = 0
                else:
                    score_percentage = "%s%s" %(score_percentage,"%")
                pred_detail["score"] = score_percentage
                pred_details.append(pred_detail)
                counter+=1
        except:
           pred_details={}
        
        predicted_topic_id = y_predict[0]
        predicted_topic = self.get_topic_name(id=predicted_topic_id) 
        return predicted_topic,pred_details

