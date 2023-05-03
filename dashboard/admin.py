from django.contrib import admin
from .models import CountryData, EmotionsData

# Register your models here.


admin.site.register(CountryData)
admin.site.register(EmotionsData)
