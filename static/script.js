document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Função para adicionar uma mensagem ao chat
    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Rola para a última mensagem
    }

    // Função para enviar uma mensagem
    function sendMessage() {
        const message = userInput.value.trim();

        if (!message) {
            alert('Por favor, digite uma mensagem.');
            return;
        }

        // Adiciona a mensagem do usuário ao chat
        appendMessage('user', message);
        userInput.value = ''; // Limpa o campo de entrada

        // Exibe o indicador de carregamento
        loadingIndicator.style.display = 'block';

        // Envia a mensagem para o servidor
        fetch('/get_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(res => {
            if (!res.ok) {
                throw new Error('Erro ao conectar com o servidor.');
            }
            return res.json();
        })
        .then(data => {
            if (data.status === 'success') {
                appendMessage('bot', data.response);
            } else {
                appendMessage('bot', 'Erro: ' + data.error);
            }
        })
        .catch(error => {
            appendMessage('bot', 'Erro ao conectar com o servidor.');
            console.error('Erro:', error);
        })
        .finally(() => {
            // Oculta o indicador de carregamento
            loadingIndicator.style.display = 'none';
        });
    }

    // Evento de clique no botão de enviar
    sendButton.addEventListener('click', sendMessage);

    // Evento de pressionar Enter no campo de entrada
    userInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    // Função para limpar o chat
    function clearChat() {
        chatBox.innerHTML = ''; // Remove todas as mensagens
    }

    // Botão para limpar o chat (se existir)
    const clearButton = document.getElementById('clear-button');
    if (clearButton) {
        clearButton.addEventListener('click', clearChat);
    }

    // Função para rolar automaticamente para a última mensagem
    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Rola para o final do chat ao carregar a página
    scrollToBottom();
});
