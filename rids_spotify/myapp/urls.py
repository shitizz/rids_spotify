from django.urls import path
from .views import base,login
urlpatterns = [
    path('', base, name='base'),
    path('login/',login,name="login")  # Use an appropriate URL path
]
