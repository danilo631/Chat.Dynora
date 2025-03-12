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
    async function sendMessage() {
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

        try {
            // Envia a mensagem para o servidor
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`Erro no servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.status === 'success') {
                appendMessage('bot', data.response);
            } else {
                appendMessage('bot', `Erro: ${data.error || 'Resposta inválida do servidor.'}`);
            }
        } catch (error) {
            console.error('Erro:', error);
            appendMessage('bot', `Erro ao conectar com o servidor: ${error.message}`);
        } finally {
            // Oculta o indicador de carregamento
            loadingIndicator.style.display = 'none';
        }
    }

    // Evento de clique no botão de enviar
    sendButton.addEventListener('click', sendMessage);

    // Evento de pressionar Enter no campo de entrada
    userInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    // Rola para o final do chat ao carregar a página
    chatBox.scrollTop = chatBox.scrollHeight;
});
