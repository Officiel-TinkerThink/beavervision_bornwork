# app/api/__init__.py
from fastapi import APIRouter

# Create a router instance that will be imported by main.py
router = APIRouter()

# Import all endpoints to register them with the router
from .endpoints import *