document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function sendMessage() {
        const message = userInput.value.trim();

        if (!message) return;

        appendMessage('user', message);
        userInput.value = '';

        const thinking = document.createElement('div');
        thinking.className = 'message bot';
        thinking.textContent = 'DYNORA está pensando...';
        chatBox.appendChild(thinking);

        fetch('/get_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(res => res.json())
        .then(data => {
            thinking.remove();
            appendMessage('bot', data.response);
        })
        .catch(() => {
            thinking.remove();
            appendMessage('bot', 'Erro ao conectar com o servidor.');
        });
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });
});
