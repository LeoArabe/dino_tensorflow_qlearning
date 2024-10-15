// test.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const puppeteer = require('puppeteer');
const fs = require('fs');

const app = express();
app.use(express.static(path.join(__dirname, 'public-test')));

// Função para carregar os modelos
async function loadModel(generation) {
    const modelPath = `file://./models/generation_${generation}/model.json`;
    try {
        const model = await tf.loadLayersModel(modelPath);
        console.log(`Modelo da geração ${generation} carregado com sucesso.`);
        return model;
    } catch (error) {
        console.error(`Erro ao carregar o modelo da geração ${generation}:`, error);
        return null;
    }
}

// Função para normalizar o estado do jogo
function normalizeState(state) {
    const maxDistance = 600;
    const maxTReXJumpHeight = 100;
    const maxVelocityY = 20;

    const distanceToObstacle = state.obstacleX !== null
        ? (state.obstacleX - state.tRexX) / maxDistance
        : 1; // 1 indica que não há obstáculo próximo

    const obstacleInSight = state.obstacleX !== null ? 1 : 0;
    const tRexOnGround = state.tRexY === 0 ? 1 : 0;

    return [
        (state.tRexY + maxTReXJumpHeight) / maxTReXJumpHeight,
        state.tRexVY / maxVelocityY,
        distanceToObstacle,
        (state.obstacleWidth || 0) / 50,
        (state.obstacleHeight || 0) / 50,
        obstacleInSight,
        tRexOnGround,
    ];
}

// Função para escolher a ação usando o modelo
function chooseAction(state, model) {
    const normalizedState = normalizeState(state);
    const stateTensor = tf.tensor2d([normalizedState]);
    const actionProbs = model.predict(stateTensor);
    const action = actionProbs.argMax(1).dataSync()[0];
    tf.dispose([stateTensor, actionProbs]);
    return action;
}

// Função para iniciar um novo servidor em uma porta específica
function startServer(generation, model) {
    const app = express();
    app.use(express.static(path.join(__dirname, 'public-test')));
    const server = http.createServer(app);
    const io = socketIo(server);

    io.on('connection', (socket) => {
        console.log(`Novo cliente conectado para a geração ${generation}`);
        socket.emit('startGame');

        socket.on('state', (gameState) => {
            const state = {
                tRexY: gameState.dinoY,
                tRexVY: gameState.dinoVY,
                tRexX: gameState.dinoX,
                obstacleX: gameState.obstacleX,
                obstacleWidth: gameState.obstacleWidth,
                obstacleHeight: gameState.obstacleHeight,
            };

            if (state.obstacleX === null || state.obstacleX === undefined) {
                state.obstacleX = 600; // Definir como distância máxima
            }

            const action = chooseAction(state, model);
            socket.emit('action', action);
        });

        socket.on('disconnect', () => {
            console.log(`Cliente desconectado da geração ${generation}`);
        });
    });

    const port = 3001 + generation;
    server.listen(port, () => {
        console.log(`Servidor para a geração ${generation} rodando na porta ${port}`);
    });

    openTestDinoGame(port);
}

// Função para abrir o jogo Dino em uma nova janela
async function openTestDinoGame(port) {
    const screenWidth = 900;
    const screenHeight = 900;

    const windowWidth = Math.floor(screenWidth * 0.8);
    const windowHeight = Math.floor(screenHeight / 3);

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
    await page.goto(`http://localhost:${port}`);

    await page.evaluateOnNewDocument(() => {
        Object.defineProperty(document, 'hidden', { value: false });
        Object.defineProperty(document, 'visibilityState', { value: 'visible' });
        document.addEventListener('visibilitychange', (event) => {
            event.stopImmediatePropagation();
        }, true);
    });
}

// Função para carregar os modelos e iniciar os servidores
async function init() {
    // Verificar quantas gerações foram salvas
    const generations = [];
    const modelDir = './models/';
    if (fs.existsSync(modelDir)) {
        const dirs = fs.readdirSync(modelDir, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);

        dirs.forEach(dir => {
            const match = dir.match(/generation_(\d+)/);
            if (match) {
                generations.push(parseInt(match[1]));
            }
        });

        // Ordenar as gerações
        generations.sort((a, b) => a - b);

        // Iniciar servidores para cada modelo disponível
        for (let i = 0; i < generations.length; i++) {
            const generation = generations[i];
            const model = await loadModel(generation);
            if (model) {
                startServer(generation, model);
            }
        }
    } else {
        console.error('Pasta de modelos não encontrada.');
    }
}

init();
