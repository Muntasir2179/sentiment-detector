from django.db import models

# Create your models here.


class CountryData(models.Model):
    country = models.CharField(max_length=100)
    population = models.IntegerField()

    class Meta:
        verbose_name_plural = 'Country Population Data'

    def __str__(self):
        return f'{self.country}-{self.population}'


class EmotionsData(models.Model):
    text = models.TextField(max_length=500)
    sadness = models.FloatField()
    joy = models.FloatField()
    anger = models.FloatField()
    love = models.FloatField()
    surprise = models.FloatField()
    fear = models.FloatField()

    class Meta:
        verbose_name_plural = 'Emotion Data'

    def __str__(self):
        return f'{self.text}'
