from django.urls import path, include
import django
from . import views

urlpatterns = [ path('home/',views.home, name='CRM-home')]#,path('accounts/login/',views.login, name='CRM-user-auth')]

# (r'^login/$', 'django.contrib.auth.views.login', {'template_name': 'login.html'}),
#                (r'^logout/$','django.contrib.auth.views.logout', {'next_page': '/login/'})