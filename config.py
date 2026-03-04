"""
Configuration Module
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    DEBUG = True
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    ALLOWED_EXTENSIONS = {'txt'}
    MODELS_FOLDER = 'models'
    LOG_FOLDER = 'logs'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True