from django.shortcuts import render, HttpResponse

from django.conf import settings

from django.core import serializers
from django.http import JsonResponse
from fastai.text.all import *
import threading
learn = load_learner(settings.BASE_DIR/'textgenerator/static/textgenerator/export1.pkl')
lock = threading.Lock()
# Create your views here.
def index(request):
    return render(request, 'textgenerator/index.html');

waitt = [True]
#thread = threading.Thread(target=background_calculation)
def text(request):
    
    TEXT = request.POST["text"]
    N_WORDS = int(request.POST["words"])
    with lock:
        preds =learn.predict(TEXT, N_WORDS, temperature=0.75)
        return JsonResponse({"instance": preds}, status=200)
    return JsonResponse({"instance": "Something went wrong"}, status=404)