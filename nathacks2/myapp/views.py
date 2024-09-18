from django.shortcuts import render
from .models import Chat
from django.shortcuts import redirect
from django.http import JsonResponse
import openai

from django.contrib import auth
from django.contrib.auth.models import User
# from .models import Chat

from django.utils import timezone

OPENAI_API_KEY = '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'
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(message):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a trained psychotherapist, specializing in providing stress management strategies for people with ADHD. Give short responses for every query, less than 4 sentences. "},
            {"role": "user", "content": message}
        ]
    )
    
    response = completion.choices[0].message.content
    return response

def home(request):
    
    if request.method == 'POST': # If user sends a message
        message = request.POST.get('message')
        response = "This is a response"
        response = ask_openai(message)
        # chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        # chat.save()
        return JsonResponse({'message': message, 'response': response})
        
    
    return render(request, 'myapp/home.html')

def webcam_view(request):
    return render(request, 'myapp/webcam.html')