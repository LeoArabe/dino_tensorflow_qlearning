// Server // Importações e Configurações Iniciais
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const puppeteer = require('puppeteer');

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const io = socketIo(server);

const port = 3001;

// Variável para o modelo
let model;

// Função para carregar o modelo
async function loadModel() {
    try {
        const loadedModel = await tf.loadLayersModel('file://./model/model.json');
        console.log('Modelo carregado com sucesso');
        // Não é necessário compilar o modelo para inferência
        // loadedModel.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
        return loadedModel;
    } catch (error) {
        console.error('Erro ao carregar o modelo:', error);
    }
}

// Função para escolher a ação usando o modelo
function chooseAction(state, model, epsilon) {
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * 3); // Ação aleatória
    } else {
        const stateTensor = tf.tensor2d([state], [1, 5]);
        const qValues = model.predict(stateTensor);
        const action = qValues.argMax(1).dataSync()[0];
        stateTensor.dispose();
        qValues.dispose();
        return action;
    }
}

// Evento de conexão do Socket.IO
io.on('connection', (socket) => {
    console.log('Novo cliente conectado');
    socket.emit('startGame');

    // Evento para receber o estado do jogo
    socket.on('state', (gameState) => {
        // Extrair o estado atual
        const currentState = [
            gameState.dinoState,
            gameState.distObstaculo,
            gameState.tamObstaculo,
            gameState.velJogo,
            gameState.scoreAtual,
        ];

        if (currentState[1] === null) {
            currentState[1] = 0;
        }

        if (gameState.gameOver) {
            socket.emit('reset'); // Notificar o cliente para reiniciar o jogo
            return;
        }

        // Escolher a ação usando o modelo treinado
        const action = chooseAction(currentState, model, 0); // epsilon = 0

        // Enviar a ação escolhida para o cliente
        socket.emit('action', action);
    });

    socket.on('disconnect', () => {
        console.log('Cliente desconectado');
    });
});


async function openTestDinoGames() {
    const screenWidth = 900; // Largura da tela padrão, ajuste conforme necessário
    const screenHeight = 900; // Altura da tela padrão, ajuste conforme necessário

    const windowWidth = Math.floor(screenWidth * 0.8); // 20% menor em largura
    const windowHeight = Math.floor(screenHeight / 3); // 3x menor em altura

        const browser = await puppeteer.launch({
            headless: false,
            args: [
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                `--window-size=${windowWidth},${windowHeight}`
            ],
        });
        const page = await browser.newPage();
        await page.goto('http://localhost:3001');

        // Desabilitar o throttling na página
        await page.evaluateOnNewDocument(() => {
            Object.defineProperty(document, 'hidden', { value: false });
            Object.defineProperty(document, 'visibilityState', { value: 'visible' });
            document.addEventListener('visibilitychange', (event) => {
                event.stopImmediatePropagation();
            }, true);
        });
    }


// Função para carregar o modelo e iniciar o servidor
async function init() {
    model = await loadModel(); // Carrega o modelo treinado
    server.listen(port, () => {
        console.log(`Servidor rodando na porta ${port}`);
        // Você pode abrir manualmente o jogo no navegador em http://localhost:3000
    });
    openTestDinoGames()
}

init();
