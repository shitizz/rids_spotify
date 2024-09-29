from django import forms
from .models import UserRegistration

class UserRegistrationForm(forms.ModelForm):
    class Meta:
        model = UserRegistration
        fields = ['username', 'email', 'password']
        widgets = {
            'password': forms.PasswordInput(),  # Use a password input for security
        }
    
    def clean_username(self):
        username = self.cleaned_data.get('username')
        if UserRegistration.objects.filter(username=username).exists():
            raise forms.ValidationError("Username is already taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if UserRegistration.objects.filter(email=email).exists():
            raise forms.ValidationError("Email is already in use.")
        return email


