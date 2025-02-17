from flask import Flask
from flask_socketio import SocketIO
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Create the SocketIO instance
socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    
    # Load the SECRET_KEY from the environment variable
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

    # If the SECRET_KEY is not set in the environment, provide a fallback (for development purposes)
    if not app.config['SECRET_KEY']:
        app.config['SECRET_KEY'] = 'default_secret_key'  # Use this in development only, replace for production

    socketio.init_app(app)
    return app
