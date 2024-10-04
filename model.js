// Server // Importações e Configurações Iniciais
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node-gpu');
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

console.log('Backend TensorFlow-GPU:', tf.getBackend());

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

// Variáveis para armazenar as métricas
const trainingLosses = [];
const topScoresOverTime = [];
const agentsPerformance = {};

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
        qValues.dispose();
        return action;
    }
}

// Evento de conexão do Socket.IO
io.on('connection', (socket) => {
    console.log(`Novo cliente conectado`);
    socket.emit('startGame');

    agentsPerformance[socket.id] = {
        score: 0,
        maxScore: 0
    };

    // Variáveis para armazenar o último estado e ação
    let lastState = null;
    let lastAction = null;

    // Evento para receber o estado do jogo
    socket.on('state', async (gameState) => {
        // Atualizar a pontuação do agente
        agentsPerformance[socket.id].score = gameState.scoreAtual;

        if (gameState.scoreAtual > agentsPerformance[socket.id].maxScore) {
            agentsPerformance[socket.id].maxScore = gameState.scoreAtual;

            const topScoreEntry = {
                time: Date.now(),
                agentId: socket.id,
                score: gameState.scoreAtual
            };
            topScoresOverTime.push(topScoreEntry);

            // Emitir o novo top score
            io.emit('topScoreUpdate', topScoreEntry);

            // Limitar o tamanho de topScoresOverTime
            if (topScoresOverTime.length > 1000) {
                topScoresOverTime.shift();
            }
        }

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

        // Escolher a ação usando epsilon-greedy
        const action = chooseAction(currentState, model, epsilon);

        // Enviar a ação escolhida para o cliente
        socket.emit('action', action);

        // Se havia um estado anterior, armazenar a experiência no replay buffer
        if (lastState !== null && lastAction !== null) {
            // Definir a recompensa
            const reward = gameState.gameOver ? -100 : 1; // Penalidade ao colidir, recompensa por sobreviver

            const agentPerformance = agentsPerformance[socket.id];
            const priority = agentPerformance.maxScore;

            replayBuffer.push({
                agentId: socket.id,
                state: lastState,
                action: lastAction,
                reward,
                nextState: currentState,
                done: gameState.gameOver,
                priority: priority // Inclua a prioridade aqui
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
        delete agentsPerformance[socket.id];
    });
});

// Função para amostrar experiências com base na prioridade
function sampleExperiences(batchSize) {
    // Calcular a soma das prioridades
    const totalPriority = replayBuffer.reduce((sum, exp) => sum + exp.priority, 0);

    const batch = [];
    for (let i = 0; i < batchSize; i++) {
        const rand = Math.random() * totalPriority;
        let cumulative = 0;
        for (const exp of replayBuffer) {
            cumulative += exp.priority;
            if (cumulative >= rand) {
                batch.push(exp);
                break;
            }
        }
    }
    return batch;
}

// Treinar o modelo e salvá-lo periodicamente usando experiências do replay buffer
setInterval(async () => {
    if (replayBuffer.length >= BATCH_SIZE) {
        // Amostrar um minibatch com base nas prioridades
        const batch = sampleExperiences(BATCH_SIZE);

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
        const history = await model.fit(statesTensor, targetsTensor, { epochs: 1, verbose: 0 });

        // Armazenar a perda (loss)
        const loss = history.history.loss[0];
        trainingLosses.push({
            step: trainingStep,
            loss: loss
        });

        // Emitir o novo valor de perda
        io.emit('trainingLossUpdate', {
            step: trainingStep,
            loss: loss
        });

        // Limitar o tamanho de trainingLosses
        if (trainingLosses.length > 1000) {
            trainingLosses.shift();
        }

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
            executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', // Caminho do Chrome instalado
            headless: true,
            args: [
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--enable-gpu-rasterization',
                '--enable-oop-rasterization',
                '--enable-accelerated-2d-canvas',
                '--enable-webgl',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--enable-features=HeadlessMode',
                '--disable-software-rasterizer',
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

// Endpoints da API para fornecer os dados
app.get('/api/training-loss', (req, res) => {
    res.json(trainingLosses);
});

app.get('/api/top-scores', (req, res) => {
    res.json(topScoresOverTime);
});

// Função para abrir o dashboard no navegador
function openDashboard() {
    const dashboardUrl = `http://localhost:${port}/dashboard.html`;

    // Abrir a URL no navegador padrão
    let startCommand;

    switch (process.platform) {
        case 'darwin':
            startCommand = `open ${dashboardUrl}`;
            break;
        case 'win32':
            startCommand = `start ${dashboardUrl}`;
            break;
        case 'linux':
            startCommand = `xdg-open ${dashboardUrl}`;
            break;
        default:
            console.log(`Por favor, abra manualmente o dashboard em: ${dashboardUrl}`);
            return;
    }

    exec(startCommand, (error) => {
        if (error) {
            console.error(`Erro ao abrir o dashboard: ${error}`);
        } else {
            console.log(`Dashboard aberto em ${dashboardUrl}`);
        }
    });
}

// Função para carregar o modelo e iniciar o servidor
async function init() {
    model = await loadModel(); // Certifique-se de que o modelo está carregado
    server.listen(port, () => {
        console.log(`Servidor rodando na porta ${port}`);

        // Abrir o dashboard
        openDashboard();

        // Aguardar alguns segundos antes de abrir os jogos
        setTimeout(() => {
            openMultipleDinoGames(100); // Abre 40 instâncias do jogo
        }, 5000); // Aguarda 5 segundos
    });
}

init();