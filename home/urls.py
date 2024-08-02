from unicodedata import name
from django.urls import path
from .views import video_feed, index

urlpatterns = [
    path('', index),
    path('video_feed/', video_feed, name="video-feed"),
]