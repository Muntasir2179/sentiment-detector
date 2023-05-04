from django.shortcuts import render
from .models import EmotionsData
import joblib

# Create your views here.

# loading the trained model
pipe_lr = joblib.load(
    open('models/emotion_classifier_pipeline_lr_28_april_2023.pkl', 'rb'))


def index(request):
    return render(request, 'dashboard/index.html')


def make_prediction(request):
    return render(request, 'dashboard/make_prediction.html')


def prediction_history(request):
    return render(request, 'dashboard/prediction_history.html')


def prediction(request):
    text = ''
    if request.method == 'POST':
        text = request.POST.get('text')
    prediction = pipe_lr.predict([text])
    data = pipe_lr.predict_proba([text])
    labels = pipe_lr.classes_
    probability = data.tolist()[0]

    # saving predictions to the database
    emotion = EmotionsData()
    emotion.text = text
    emotion.anger = probability[0]
    emotion.fear = probability[1]
    emotion.joy = probability[2]
    emotion.love = probability[3]
    emotion.sadness = probability[4]
    emotion.surprise = probability[5]
    emotion.save()

    context = {
        'labels': labels,
        'probability': probability,
    }
    return render(request, 'dashboard/prediction.html', context)
