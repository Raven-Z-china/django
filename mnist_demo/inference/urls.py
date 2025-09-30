from django.urls import path
from . import views

urlpatterns = [
    path('', views.mnist_inference, name='mnist_inference'),
]