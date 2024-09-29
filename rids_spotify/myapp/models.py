# Create your models here.
# myapp/models.py

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db import models

class UserRegistrationManager(BaseUserManager):
    def create_user(self, username, email, password=None):
        if not email:
            raise ValueError("Users must have an email address")
        if not username:
            raise ValueError("Users must have a username")
        
        user = self.model(
            username=username,
            email=self.normalize_email(email),
        )
        
        user.set_password(password)  # Use set_password to hash the password
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None):
        user = self.create_user(username, email, password)
        user.is_admin = True
        user.is_staff = True
        user.save(using=self._db)
        return user

class UserRegistration(AbstractBaseUser):
    user_id = models.AutoField(primary_key=True)  # Custom primary key
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Store hashed passwords

    # Required fields for user creation
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']

    objects = UserRegistrationManager()

    def __str__(self):
        return self.username

