// worker.js
const { workerData, parentPort } = require('worker_threads');
const GameEngine = require('./gameEngine');
const tf = require('@tensorflow/tfjs-node-gpu');

// Dados recebidos do server.js
const workerId = workerData.workerId;
const numInstances = workerData.numInstances || 1;
let modelWeights = workerData.modelWeights; // Pesos recebidos do servidor
const currentGeneration = workerData.currentGeneration;
const maxGenerations = workerData.maxGenerations;

// Array para armazenar as instâncias do jogo
const gameEngines = [];

// Inicializar as instâncias do jogo
for (let i = 0; i < numInstances; i++) {
  gameEngines.push(new GameEngine());
}

// Função para criar o modelo
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 128,
    inputShape: [7],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
  }));
  return model;
}

let model;

// Carregar o modelo com os pesos fornecidos
async function loadModel() {
  model = createModel();
  if (modelWeights) {
    const weightTensors = modelWeights.map(w => tf.tensor(w, undefined, 'float32'));
    model.setWeights(weightTensors);
  } else {
    // Inicializar pesos aleatoriamente se não forem fornecidos
    console.log(`Worker ${workerId}: Inicializando modelo com pesos aleatórios`);
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
    state.tRexVelocityY / maxVelocityY,
    distanceToObstacle,
    (state.obstacleWidth || 0) / 50,
    (state.obstacleHeight || 0) / 50,
    obstacleInSight,
    tRexOnGround,
  ];
}

// Função para escolher a ação usando o modelo
function chooseAction(state) {
  const normalizedState = normalizeState(state);
  const stateTensor = tf.tensor2d([normalizedState]);
  const actionProbs = model.predict(stateTensor);
  const action = actionProbs.argMax(1).dataSync()[0];
  tf.dispose([stateTensor, actionProbs]);
  return action;
}

// Função para executar o jogo e calcular o fitness
async function runGames() {
  let totalScore = 0;
  const numGames = gameEngines.length;

  for (let i = 0; i < numGames; i++) {
    let gameEngine = gameEngines[i];

    // Reiniciar o jogo
    gameEngine.reset();

    // Executar o jogo até o fim
    while (!gameEngine.gameOver) {
      const state = gameEngine.getState();
      const action = chooseAction(state);
      gameEngine.update(action);
    }

    totalScore += gameEngine.score;
  }

  // Calcular a pontuação média como fitness
  const averageScore = totalScore / numGames;

  // Enviar o fitness de volta ao servidor
  parentPort.postMessage({
    type: 'fitness',
    workerId: workerId,
    fitness: averageScore,
  });
}

// Iniciar o worker
async function init() {
  await loadModel();

  // Executar a primeira vez
  runGames();
}

init();
