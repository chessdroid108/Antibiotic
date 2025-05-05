import os
import logging
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define base model class
class Base(DeclarativeBase):
    pass

# Initialize db
db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_for_testing")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Load configuration
app.config.from_pyfile('config.py')

# Initialize the app with the SQLAlchemy extension
db.init_app(app)

# Register context processors
@app.context_processor
def utility_processor():
    return {
        'now': datetime.utcnow
    }

# Import routes and register them
with app.app_context():
    # Import models to ensure they're registered with SQLAlchemy
    from models import User, Project, GenomeData, Result
    
    # Create database tables if they don't exist
    db.create_all()
    
    # Register routes
    from routes import register_routes
    register_routes(app)

logger.info("Application initialized successfully")
