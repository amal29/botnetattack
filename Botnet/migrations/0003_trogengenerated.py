# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2018-04-11 11:16
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Botnet', '0002_attacks'),
    ]

    operations = [
        migrations.CreateModel(
            name='Trogengenerated',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=10)),
                ('ip', models.CharField(max_length=40)),
                ('status', models.CharField(max_length=10)),
            ],
        ),
    ]
