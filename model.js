// Server // Importações e Configurações Iniciais
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node');
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const io = socketIo(server);

const port = 3000;

// Configurações de Q-Learning e Variáveis Globais
let epsilon = 1.0;
const epsilonDecay = 0.995;
const epsilonMin = 0.01;
const gamma = 0.95;

const replayBuffer = [];
const MAX_BUFFER_SIZE = 100000; // Capacidade máxima do buffer

const BATCH_SIZE = 64;          // Tamanho do minibatch
const TRAINING_INTERVAL = 100;  // Intervalo de treinamento em milissegundos
let trainingStep = 0;           // Variável para contar os passos de treinamento
const SAVE_MODEL_INTERVAL = 1000; // Salvar o modelo a cada 1000 passos de treinamento (ajuste conforme necessário)


let model;

// Funções de Q-Learning (criar modelo, escolher ação, treinar modelo)
function createDinoModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 24, inputShape: [5], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    return model;
}

// Função para carregar o modelo
async function loadModel() {
    try {
        const loadedModel = await tf.loadLayersModel('file://./model/model.json');
        console.log('Modelo carregado com sucesso');

        // Compilar o modelo após carregá-lo
        loadedModel.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

        return loadedModel;
    } catch (error) {
        console.error('Erro ao carregar o modelo:', error);
        const newModel = createDinoModel();
        console.log('Novo modelo criado.');
        return newModel;
    }
}


// Função para salvar o modelo
async function saveModel(model) {
    try {
        await model.save('file://./model');
        console.log('Modelo salvo com sucesso');
    } catch (error) {
        console.error('Erro ao salvar o modelo:', error);
    }
}

// Função para escolher a ação usando epsilon-greedy
function chooseAction(state, model, epsilon) {
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * 3); // 0 = pular, 1 = abaixar, 2 = correr
    } else {
        const stateTensor = tf.tensor2d([state], [1, 5]);
        const qValues = model.predict(stateTensor);
        const action = qValues.argMax(1).dataSync()[0];
        stateTensor.dispose();
        return action;
    }
}

// Evento de conexão do Socket.IO
io.on('connection', (socket) => {
    console.log('Novo cliente conectado');
    socket.emit('startGame');

    // Variáveis para armazenar o último estado e ação
    let lastState = null;
    let lastAction = null;

    // Evento para receber o estado do jogo
    socket.on('state', async (gameState) => {
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
            // Atualizar epsilon após o fim do jogo
            epsilon = Math.max(epsilon * epsilonDecay, epsilonMin);
            socket.emit('reset'); // Notificar o cliente para reiniciar o jogo

            // Reiniciar o estado
            lastState = null;
            lastAction = null;
            return;
        }

        let scoreReal = Math.round(currentState[4]*0.025);
        // Escolher a ação usando epsilon-greedy
        const action = chooseAction(currentState, model, epsilon);
        if(scoreReal > 90){
            console.log('Current State:', scoreReal);
        }
        //console.log('Chosen Action:', action);

        // Enviar a ação escolhida para o cliente
        socket.emit('action', action);

        // Se havia um estado anterior, armazenar a experiência no replay buffer
        if (lastState !== null && lastAction !== null) {
            // Definir a recompensa
            const reward = gameState.gameOver ? -100 : 1; // Penalidade ao colidir, recompensa por sobreviver

            replayBuffer.push({
                state: lastState,
                action: lastAction,
                reward: reward,
                nextState: currentState,
                done: gameState.gameOver
            });

            // Limitar o tamanho do buffer
            if (replayBuffer.length > MAX_BUFFER_SIZE) {
                replayBuffer.shift();
            }
        }

        // Atualizar o estado anterior e ação anterior
        lastState = currentState;
        lastAction = action;
    });

    socket.on('disconnect', () => {
        console.log('Cliente desconectado');
    });
});

// Treinar o modelo e salvá-lo periodicamente usando experiências do replay buffer
setInterval(async () => {
    if (replayBuffer.length >= BATCH_SIZE) {
        // Amostrar um minibatch aleatório
        const batch = [];
        for (let i = 0; i < BATCH_SIZE; i++) {
            const index = Math.floor(Math.random() * replayBuffer.length);
            batch.push(replayBuffer[index]);
        }

        // Preparar os dados para treinamento
        const states = batch.map(e => e.state);
        const actions = batch.map(e => e.action);
        const rewards = batch.map(e => e.reward);
        const nextStates = batch.map(e => e.nextState);
        const dones = batch.map(e => e.done);

        // Converter para tensores
        const statesTensor = tf.tensor2d(states);
        const nextStatesTensor = tf.tensor2d(nextStates);

        // Predizer Q-values atuais e próximos
        const qValues = model.predict(statesTensor);
        const qNextValues = model.predict(nextStatesTensor);

        // Obter os valores em arrays para manipulação
        const qValuesArray = qValues.arraySync();
        const qNextValuesArray = qNextValues.arraySync();

        for (let i = 0; i < batch.length; i++) {
            const target = rewards[i] + (gamma * Math.max(...qNextValuesArray[i]) * (1 - dones[i]));
            qValuesArray[i][actions[i]] = target;
        }

        // Treinar o modelo
        const targetsTensor = tf.tensor2d(qValuesArray);
        await model.fit(statesTensor, targetsTensor, { epochs: 1, verbose: 0 });

        // Liberar memória
        statesTensor.dispose();
        nextStatesTensor.dispose();
        targetsTensor.dispose();
        qValues.dispose();
        qNextValues.dispose();

        // Incrementar o contador de passos de treinamento
        trainingStep++;

        // Salvar o modelo a cada N passos de treinamento
        if (trainingStep % SAVE_MODEL_INTERVAL === 0) {
            await saveModel(model);
            console.log(`Modelo salvo no passo de treinamento ${trainingStep}`);
        }
    }
}, TRAINING_INTERVAL); // Exemplo: a cada 100 ms

async function openMultipleDinoGames(numInstances) {
    const screenWidth = 900; // Largura da tela padrão, ajuste conforme necessário
    const screenHeight = 900; // Altura da tela padrão, ajuste conforme necessário

    const windowWidth = Math.floor(screenWidth * 0.8); // 20% menor em largura
    const windowHeight = Math.floor(screenHeight / 3); // 3x menor em altura

    for (let i = 0; i < numInstances; i++) {
        const browser = await puppeteer.launch({
            headless: true,
            args: [
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                `--window-size=${windowWidth},${windowHeight}`
            ],
        });
        const page = await browser.newPage();
        await page.goto('http://localhost:3000');

        // Desabilitar o throttling na página
        await page.evaluateOnNewDocument(() => {
            Object.defineProperty(document, 'hidden', { value: false });
            Object.defineProperty(document, 'visibilityState', { value: 'visible' });
            document.addEventListener('visibilitychange', (event) => {
                event.stopImmediatePropagation();
            }, true);
        });
    }
}


// Função para carregar o modelo e iniciar o servidor
async function init() {
    model = await loadModel(); // Certifique-se de que o modelo está carregado
    server.listen(port, () => {
        console.log(`Servidor rodando na porta ${port}`);
        openMultipleDinoGames(50); // Abre 5 instâncias do jogo
    });
}

init();