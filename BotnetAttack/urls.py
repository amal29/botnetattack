"""BotnetAttack URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from Botnet.views import *

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$',firstpg),
    url(r'^loginpg',loginpg,name="loginpg"),
    url(r'^home',home,name="home"),
    url(r'^gallery',gallery,name="gallery"),
    url(r'^about',about,name="about"),
    url(r'^reguser',reguser,name="reguser"),
    url(r'^loginreq',loginreq,name="loginreq"),
    url(r'^athome',athome,name="athome"),
    url(r'^logout',logout,name="logout"),
    url(r'^sendattacks',sendattacks,name="sendattacks"),
    url(r'^sendattacktouser',sendattacktouser,name="sendattacktouser"),
    url(r'^userhome',userhome,name="userhome"),
    url(r'^incommingreq',incommingreq,name="incommingreq"),
    url(r'^trainpg',trainpg,name="trainpg"),
    url(r'^analysis',analysis,name="analysis"),
    url(r'^train',train,name="train"),
    url(r'^svmtrainingdata',svmtrainingdata,name="svmtrainingdata"),
    url(r'^testattack',testattack,name="testattack"),
    url(r'^generateattacks',generateattacks,name="generateattacks"),
    url(r'^svmtesting',svmtesting,name="svmtesting"),
    url(r'^accana',accana,name="accana"),
    url(r'^timeana',timeana,name="timeana"),
    url(r'^genatt',genatt,name="genatt")
]

