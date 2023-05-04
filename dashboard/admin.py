from django.contrib import admin
from .models import EmotionsData

# Register your models here.


class EmotionAdmin(admin.ModelAdmin):
    list_display = ('text', 'prediction', 'prediction_confidence',
                    'user_prediction', 'is_correct_prediction')
    list_display_links = ('text', 'prediction', 'user_prediction')


admin.site.register(EmotionsData, EmotionAdmin)
