const socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', () => {
    console.log('Connected to WebSocket');
});

socket.on('message', (data) => {
    const messages = document.getElementById('messages');
    const messageElement = document.createElement('div');
    messageElement.textContent = data;
    messages.appendChild(messageElement);
    messages.scrollTop = messages.scrollHeight;  // Scroll to bottom
});

function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value;
    if (message) {
        socket.send(message);  // Send message to server
        messageInput.value = '';  // Clear input field
    }
}
