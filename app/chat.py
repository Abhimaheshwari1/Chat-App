from flask import render_template
from app import socketio

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

