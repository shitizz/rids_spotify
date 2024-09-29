from django.shortcuts import render, HttpResponse, redirect
from .models import UserRegistration
from django.contrib.auth.hashers import make_password

# Create your views here.
def base(request):
    return render(request, 'rids/base.html')
""""
def login(request):
    return render(request,'rids/login.html')
"""

from django.shortcuts import render, redirect
from .forms import UserRegistrationForm

def login(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])  # Hash the password
            user.save()
            return redirect('login')  # Redirect to a success page after registration
    else:
        form = UserRegistrationForm()

    return render(request, 'rids/login.html', {'form': form})  # Pass the form to the template




    

