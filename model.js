// server.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js para Node
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

const server = http.createServer(app);
const io = socketIo(server);

const port = 3000;

// Configurações de Q-Learning
let epsilon = 1.0;
const epsilonDecay = 0.995;
const epsilonMin = 0.01;
const gamma = 0.95;

// Funções de Q-Learning (criar modelo, escolher ação, treinar modelo)
function createDinoModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 24, inputShape: [5], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    return model;
}

const model = createDinoModel();

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

// Função para treinar o modelo
async function trainQModel(state, action, reward, nextState, done) {
    const stateTensor = tf.tensor2d([state], [1, 5]);
    const nextStateTensor = tf.tensor2d([nextState], [1, 5]);

    // Predizer Q-values para o estado atual e próximo estado
    const qValues = model.predict(stateTensor).arraySync();
    const qNextValues = model.predict(nextStateTensor).arraySync();

    // Calcular o valor alvo
    const target = reward + (gamma * Math.max(...qNextValues[0]) * (1 - done));

    // Atualizar o Q-value para a ação tomada
    qValues[0][action] = target;

    const targetTensor = tf.tensor2d(qValues);

    // Treinar o modelo com os novos valores Q
    await model.fit(stateTensor, targetTensor, { epochs: 1, verbose: 0 });

    // Liberar a memória
    stateTensor.dispose();
    nextStateTensor.dispose();
    targetTensor.dispose();
}

// Evento de conexão do Socket.IO
io.on('connection', (socket) => {
    console.log('Novo cliente conectado');

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
            gameState.scoreAtual
        ];

        if (gameState.gameOver) {
            if (lastState !== null && lastAction !== null) {
                const reward = -100; // Penalidade por colidir
                await trainQModel(lastState, lastAction, reward, currentState, true);
            }
            // Reiniciar o estado
            lastState = null;
            lastAction = null;
            epsilon = Math.max(epsilon * epsilonDecay, epsilonMin);
            socket.emit('reset'); // Notificar o cliente para reiniciar o jogo
            return;
        }

        // Escolher a ação usando epsilon-greedy
        const action = chooseAction(currentState, model, epsilon);

        // Enviar a ação escolhida para o cliente
        socket.emit('action', action);

        // Se havia um estado anterior, treinar o modelo
        if (lastState !== null && lastAction !== null) {
            const reward = gameState.scoreAtual; // Recompensa baseada no score
            await trainQModel(lastState, lastAction, reward, currentState, false);
        }

        // Atualizar o estado anterior e ação anterior
        lastState = currentState;
        lastAction = action;
    });

    socket.on('disconnect', () => {
        console.log('Cliente desconectado');
    });
});

// Iniciar o servidor
server.listen(port, () => {
    console.log(`Servidor rodando na porta ${port}`);
});
