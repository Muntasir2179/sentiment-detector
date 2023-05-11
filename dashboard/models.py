from django.db import models

# Create your models here.


class EmotionsData(models.Model):
    text = models.TextField(max_length=500)
    sadness = models.FloatField()
    joy = models.FloatField()
    anger = models.FloatField()
    love = models.FloatField()
    fear = models.FloatField()
    prediction = models.CharField(max_length=50)
    prediction_confidence = models.FloatField()
    user_prediction = models.CharField(max_length=50)
    is_correct_prediction = models.BooleanField()

    class Meta:
        verbose_name_plural = 'Emotion Data'

    def __str__(self):
        return f'{self.text}'
