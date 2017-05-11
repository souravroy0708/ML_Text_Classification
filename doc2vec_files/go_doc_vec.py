import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from textblob import TextBlob
import re
from collections import namedtuple
from os import listdir
from os.path import isfile, join
import multiprocessing
import datetime
from collections import Counter
import requests

tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words('english')

from pymongo import MongoClient
client =  MongoClient('localhost',27017)
database = client.news

class Doc_2_Vec:
    """
    get token from text
    """
    def get_token(self,text):
            #Remove all special character
            clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
            tokens = tokenizer.tokenize(clean_text)
            tokens = [token.lower() for token in tokens
                      if token not in stopwords and len(token) >= 3]
            return tokens

    def get_nouns(self,text):
        """
        Get noun from text for news keywords
        """
        noun_list = []
        text = re.sub('[^a-zA-Z0-9]', ' ',text)
        text_list = text.split()
        text_list = [token for token in text_list if len(token)>1]
        text = " ".join(text_list)
        blob = TextBlob(text)
        noun_list = list(blob.noun_phrases)
        noun_list = [keyword for keyword in noun_list if len(keyword.split())<5 ]
        # noun_list.append("|")
        # tags = blob.tags
        # tag_look = ["NNP","NN","NNS","NNPS"]
        # for tag in tags:
        #     if tag[1] in tag_look and len(tag[0])>3:
        #         noun_list.append(tag[0])
        return noun_list

    def get_regex_pattern(self,keywords):
        """
        Regex pattern to match string
        """
        base_pattern = '(?=.*{})'    
        start = '^'    
        end = '.*$'   
        patterns = []    
        for keyword in keywords:        
            pat = ''        
            if ' ' in keyword:            
                keys = keyword.split(' ')            
                for key in keys:         
                    pat += base_pattern.format(key)            
                pat = start + pat + end            
                patterns.append(pat)        
            else:            
                pat = start + base_pattern.format(keyword) + end            
                patterns.append(pat) 
            pattern = re.compile('|'.join(patterns), re.IGNORECASE)      
        return pattern

    def get_most_repeated_keyword(self,ordered_keywords=[]):
        """
        Get most repeated keyword
        """
        common_keywords_list =[]
        for common_word in ordered_keywords:
            ignore_keyword = ["based on","writer","director","romance","school","filmy","film","short","sex","lesbian","drama","government","teen","rape","abuse","3d","porn","erotic","malayalam","love","musical","animation","documentary","hindy","hindi","horror","cook","sitting"]
            regex_pattern = self.get_regex_pattern(ignore_keyword)
            if not re.match(regex_pattern,common_word):
                common_keywords_list.append(common_word)
            else:
                print("skipped keyword %s" %(common_word))
        return common_keywords_list



    def get_content_for_title(self,doc_name="",document_type="news",tmdb_id=0):
        """
        Get similar content details
        """
        description = ""
        topic = ""
        doc_name = doc_name.lower()
        try:
            record = database.doc2vec_news.find_one({"title":doc_name})
            description = record.get("description","")
            topic = record.get("topic","")
        except:
            description = ""
            print ("title does not exist.doc_name: %s document_type: %s" %(doc_name,document_type))
        return description,topic


    def get_most_similar_documet(self,document="",document_name="",document_model="model1",document_type="news",record_count=5):
        """
        Get other similar document through doc2vec
        """
        model_name = "go_doc_2_vec_%s_%s" %(document_type,document_model)

        try:
            model_saved_file = "%s" %(model_name)
            model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
        except:
            msg="Not able to load model : %s" %(model_saved_file)
        
        if document and len(document)>5:
            try:
                test_description_token = self.get_token(document)
                token_text = ' '.join(test_description_token)
                test_description_token = gensim.utils.simple_preprocess(token_text)
                infer_vector_of_document = model.infer_vector(test_description_token)
                other_similar_docs = model.docvecs.most_similar([infer_vector_of_document],topn=record_count)
                msg= "document : %s model_name : %s " %(document,model_name)
            except:
                msg="Not able to load model for get similar document for document: %s model_name : %s" %(document,model_name)

        similar_docs_list = []
        for doc in other_similar_docs:
            doc_name = doc[0]
            doc_score = doc[1]
            if doc_name !=document_name:
                similar_docs_dict = {}
                description,topic = self.get_content_for_title(doc_name=doc_name,document_type=document_type)
                similar_docs_dict["description"] = description
                similar_docs_dict["topic"] = topic
                similar_docs_dict["score"] = doc_score
                similar_docs_list.append(similar_docs_dict)
        return similar_docs_list

    #Train model and store it to reload and use it
    def train_model_and_save(self,document_type="news",document_model="model1",is_tokenize=1,m_iter=100,m_min_count=2,m_size=100,m_window=10):
            #Document Labels
            print ("Start Time : %s" %(str(datetime.datetime.now())))
            records = database.doc2vec_news.find()
            print ("Total Record count %s" %(records.count()))

            analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
            alldocuments = []
            keywords = []
            for record in records:
                title = record.get("title","")
                description = record.get("description","")
                tokens = self.get_token(description)
                #skipped less than 10 word description
                if len(tokens)<10:
                    #print ("skipped small record %s : description : %s Length: %s" %(title,description,len(tokens)))
                    continue
                words = tokens
                #words = gensim.utils.simple_preprocess(words_text)
                tags = [title]
                alldocuments.append(analyzedDocument(words, tags))

            #Train Model
            cores = multiprocessing.cpu_count()
            saved_model_name = "go_doc_2_vec_%s_%s" %(document_type,document_model)
            if document_model == "model1":
                # PV-DBOW 
                model_1 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=1)
                model_1.save("%s" %(saved_model_name))
                print ("model training completed : %s" %(saved_model_name))
            elif document_model == "model2":
                # PV-DBOW 
                model_2 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=0)
                model_2.save("%s" %(saved_model_name))
                print ("model training completed : %s" %(saved_model_name))
            elif document_model == "model3":
                # PV-DM w/average
                model_3 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_mean=1,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
                model_3.save("%s" %(saved_model_name))
                print ("model training completed : %s" %(saved_model_name))

            elif document_model == "model4":
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                model_4 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_concat=1,workers=cores, size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
                model_4.save("%s" %(saved_model_name))
                print ("model training completed : %s" %(saved_model_name))
            print ("Record count %s" %len(alldocuments))
            print ("End Time %s" %(str(datetime.datetime.now())))




