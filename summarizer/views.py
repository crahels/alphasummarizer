from django.http import HttpResponse
from django.shortcuts import render
from .modules.main import main_summarizer

def index(request):
    context = {
        'summary_type': '100',
        'topic': 'b-salah-darat',
        'model': 'best-model-10'
    }
    return render(request, 'index.html', context)

def result(request):
    summary_type = request.GET.get('summary_type')
    topic = request.GET.get('topic')
    model = request.GET.get('model')

    context = main_summarizer(int(summary_type), topic, model)

    context['summary_type'] = summary_type
    context['topic'] = topic
    context['model'] = model

    return render(request, 'result.html', context)
