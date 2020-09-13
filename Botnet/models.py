from __future__ import unicode_literals

from django.db import models

# Create your models here.

class User(models.Model):
    name = models.CharField(max_length=50)
    email = models.CharField(max_length=50)
    address = models.CharField(max_length=80)
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)

class Attacks(models.Model):
    user = models.CharField(max_length=10)
    ip = models.CharField(max_length=40)
    status = models.CharField(max_length=10)

class Trogengenerated(models.Model):
    user = models.CharField(max_length=10)
    ip = models.CharField(max_length=40)
    status = models.CharField(max_length=10)