import newspaper
from newspaper import Article
import re
import os

from pymongo import MongoClient
from datetime import datetime

client = MongoClient("localhost",27017)
database = client.news

def get_topic(url=""):
    topic = "others"
    
    sports_keyword = ["sport","sports","cricket","cricbuzz"]
    political_keyword = ["politics","political","union-budget","elections","politics"]
    entertainment_keyword = ["entertainment","celebs","bollywood"]
    technology_keyword = ["technology","tech","scitech"]
    education_keyword = ["education",]
    health_keyword = ["health","fitit","healthyliving "]
    weather_keyword = ["weather"]
    finance_keyword = ["money","finance","stocksmarkets","market","stock","mtualfund","commodity","property","personal-finance",\
                       "banking"]
    travel_keyword = ["travel"]
    business_keyword = ["business"]
    food_keyword = ["food"]
    #fashion_keyword  =["style"]
    
    
    sports_pattern = get_regex_pattern(sports_keyword)
    political_pattern = get_regex_pattern(political_keyword)
    entertainment_pattern = get_regex_pattern(entertainment_keyword)
    technology_pattern = get_regex_pattern(technology_keyword)
    education_pattern = get_regex_pattern(education_keyword)
    health_pattern = get_regex_pattern(health_keyword)
    weather_pattern = get_regex_pattern(weather_keyword)
    finance_pattern = get_regex_pattern(finance_keyword)
    travel_pattern = get_regex_pattern(travel_keyword)
    business_pattern = get_regex_pattern(business_keyword)
    food_pattern = get_regex_pattern(food_keyword)
    #fashion_pattern = get_regex_pattern(fashion_keyword)
    
    if url:
        if re.match(sports_pattern,url):
            topic = "sports"
        elif re.match(political_pattern,url):
            topic = "political"
        elif re.match(entertainment_pattern,url):
            topic = "entertainment"
        elif re.match(technology_pattern,url):
            topic = "technology"
        elif re.match(education_pattern,url):
            topic = "education"
        elif re.match(health_pattern,url):
            topic = "health"
        elif re.match(food_pattern,url):
            topic = "food"
        elif re.match(weather_pattern,url):
            topic = "weather"
        elif re.match(finance_pattern,url):
            topic = "finance"
        elif re.match(travel_pattern,url):
            topic = "travel"
        elif re.match(business_pattern,url):
            topic = "business"
        #elif re.match(fashion_pattern,url):
        #    topic = "fashion"
        else:
            topic = "others"
    return topic
        
def get_regex_pattern(keywords):
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

news_paper_domain_list = ['http://www.cnn.com',"http://www.firstpost.com",'http://www.indiatimes.com',"http://www.ndtv.com/",\
                         "http://indianexpress.com","http://www.financialexpress.com","http://economictimes.indiatimes.com/",\
                         "http://www.outlookindia.com/","http://www.dnaindia.com","http://timesofindia.indiatimes.com/",\
                         "http://www.thedailybeast.com","http://www.cricbuzz.com/","http://ewn.co.za/","https://www.forbes.com/",\
                          "http://zeenews.india.com/",\
                          "https://www.nytimes.com/","http://www.business-standard.com/","http://www.thehindu.com/"]

def scrape_news():
    counter = 0
    for news_paper_domain in news_paper_domain_list[0:]:
        news_paper = newspaper.build(news_paper_domain,memoize_articles=True,language='en')
        #news_paper = newspaper.build(news_paper_domain,language='en')
        print("%s --- %s" %(news_paper_domain,news_paper.size()))
        for paper in news_paper.articles[0:]:
            #import pdb;pdb.set_trace()
            try:
                paper.download()
                paper.parse()
                title = paper.title
                article_url = paper.url
                article_url_pattern = "/".join(article_url.split("/")[0:-1])
                content = paper.text
                content = re.sub('[^a-zA-Z0-9.\n]', ' ', content).lower()
                if len(content.split()) <50:
                    print ("skip content ---- %s" %(content))
                    continue;
                paper.nlp()
                keywords = paper.keywords
                summary  = paper.summary
                summary = re.sub('[^a-zA-Z0-9.\n]', ' ', summary).lower()
                if article_url and not "http://hindi." in article_url:
                    topic = get_topic(url = article_url_pattern)
                    if topic == "others":
                        print ("Skiped ---",topic)
                        continue;
                    title = title.lower()
                    content = content.lower()
                    current_datetime = datetime.now()
                    database.doc2vec_news.update({"title":title},{"$set":{"description":content,"title":title,"summary":summary,"keywords":keywords,"source":news_paper_domain,"topic":topic,"updated_on":current_datetime}},upsert=True)

                    counter+=1
                    print ("Topic ---  %s  url  ---  %s counter ---  %s" %(topic,article_url,counter))
                else:
                    print ("Invalid article %s" %(article_url))
            except:
                print ("Got error for article ")

scrape_news()
