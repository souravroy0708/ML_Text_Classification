from __future__ import division
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.core import urlresolvers
from django.contrib import messages
from django.contrib.auth import authenticate, login as login_auth,logout
from django.core.urlresolvers import reverse
from django.db.models import Q
from django.contrib.auth.decorators import login_required
import json
from datetime import datetime, timedelta,date
import socket
from robobrowser import RoboBrowser
from bs4 import BeautifulSoup
import re
import urllib
import random
import time
import requests
import os 

#from webapp.models import DocumentClassifier

from . import go_news_topic_classifier


import logging
logger = logging.getLogger(__name__)




def document_classifier(request):
    """
    Document topic classifier 
    """
    topic_details = {}
    doc = ""
    model = "nb"
    final_topic = ""
    # latest_result = DocumentClassifier.objects.filter().order_by("-created_date")[:5]

    if request.method == 'POST':
        doc = request.POST.get('document',"")
        model = request.POST.get('model_name',"nb")
        tp = go_news_topic_classifier.TopicClassifier()
        final_topic,topic_details = tp.get_prediction_topic(document=doc,model_name=model)
        #strore_result = DocumentClassifier.objects.create(document=doc,model=no_of_records,topic=topic_details)


    return render_to_response('googleSearch/document_classiifer.html',{"topic_details":topic_details,"doc":doc,"model":model,"final_topic":final_topic},context_instance=RequestContext(request))




