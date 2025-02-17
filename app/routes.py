from flask import render_template
from app import create_app, socketio

# Create the Flask app using the function from __init__.py
app = create_app()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('User connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('User disconnected')

@socketio.on('message')
def handle_message(data):
    print('Received message: ' + data)
    socketio.send(data, broadcast=True)
