# processor/views.py
from django.shortcuts import render
from django.conf import settings
import os

def gallery(request):
    images = []
    for f in os.listdir(settings.OUTPUT_DIR):
        if f.endswith('.png'):
            images.append(f)
    return render(request, 'gallery.html', {'images': images})