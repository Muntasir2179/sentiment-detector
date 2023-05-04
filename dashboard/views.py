from django.shortcuts import render, redirect
from .models import EmotionsData
import joblib

# Create your views here.

# loading the trained model
pipe_lr = joblib.load(
    open('trained ml model/emotion_classifier_pipeline_lr_28_april_2023.pkl', 'rb'))


def index(request):
    return render(request, 'dashboard/index.html')


def make_prediction(request):
    return render(request, 'dashboard/make_prediction.html')


def prediction_history(request):
    data = EmotionsData.objects.all()
    context = {
        'data': data,
    }
    return render(request, 'dashboard/prediction_history.html', context)


def prediction(request):
    text = ''
    if request.method == 'POST':
        text = request.POST.get('text')
    prediction = pipe_lr.predict([text])
    data = pipe_lr.predict_proba([text])
    labels = pipe_lr.classes_
    probability = data.tolist()[0]

    emotion_dict = {
        'anger': probability[0],
        'fear': probability[1],
        'joy': probability[2],
        'love': probability[3],
        'sadness': probability[4],
        'surprise': probability[5],
    }

    prediction_confidence = emotion_dict[max(
        emotion_dict, key=emotion_dict.get)]

    context = {
        'input_text': text,
        'labels': labels,
        'probability': probability,
        'prediction': prediction,
        'emotion_dict': emotion_dict,
        'prediction_confidence': prediction_confidence,
    }
    return render(request, 'dashboard/prediction.html', context)


def feedback(request):
    # saving predictions to the database
    emotion = EmotionsData()
    emotion.text = request.POST.get('input_text')
    emotion.anger = request.POST.get('anger')
    emotion.fear = request.POST.get('fear')
    emotion.joy = request.POST.get('joy')
    emotion.love = request.POST.get('love')
    emotion.sadness = request.POST.get('sadness')
    emotion.surprise = request.POST.get('surprise')

    model_prediction = request.POST.get('model_prediction')
    string = ''
    for i in model_prediction:
        if i.isalpha():
            string += i
    emotion.prediction = string

    emotion.prediction_confidence = request.POST.get('prediction_confidence')
    emotion.user_prediction = request.POST.get('user_prediction')
    emotion.is_correct_prediction = (
        string == request.POST.get('user_prediction'))
    emotion.save()

    return redirect('make-prediction')
