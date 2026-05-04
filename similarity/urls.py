from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/generate-texts/', views.generate_random_texts, name='generate_random_texts'),
]