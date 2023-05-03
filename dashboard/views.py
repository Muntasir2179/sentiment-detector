from django.shortcuts import render
from .models import CountryData, EmotionsData
import joblib

# Create your views here.

# loading the trained model
pipe_lr = joblib.load(
    open('models/emotion_classifier_pipeline_lr_28_april_2023.pkl', 'rb'))


def index(request):
    text = 'I am feeling sad'
    prediction = pipe_lr.predict([text])
    data = pipe_lr.predict_proba([text])
    labels = pipe_lr.classes_

    probability = data.tolist()[0]
    context = {
        'labels': labels,
        'probability': probability,
    }
    print(prediction)
    print(data)
    print(labels)
    return render(request, 'dashboard/index.html', context)
