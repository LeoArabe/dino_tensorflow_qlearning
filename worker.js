const { workerData, parentPort } = require('worker_threads');
const GameEngine = require('./gameEngine');
const tf = require('@tensorflow/tfjs-node-gpu');

const workerId = workerData.workerId;
const numInstances = workerData.numInstances || 1;
let modelWeights = workerData.modelWeights;
const currentGeneration = workerData.currentGeneration;
const maxGenerations = workerData.maxGenerations;

const gameEngines = [];

// Inicializar as instâncias do jogo
for (let i = 0; i < numInstances; i++) {
  gameEngines.push(new GameEngine());
}

// Função para criar o modelo
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 32,
    inputShape: [7],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: 16,
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
    try {
      const weightTensors = modelWeights.map(w => tf.tensor(w, undefined, 'float32'));
      model.setWeights(weightTensors);
    } catch (error) {
      console.error(`Erro ao carregar pesos no worker ${workerId}:`, error);
    }
  } else {
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
    : 1;

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
  return tf.tidy(() => {
    const normalizedState = normalizeState(state);
    const stateTensor = tf.tensor2d([normalizedState], undefined, 'float32');
    const actionProbs = model.predict(stateTensor);
    const action = actionProbs.argMax(1).dataSync()[0];
    return action;
  });
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

      // Enviar o estado do jogo para o servidor periodicamente
      parentPort.postMessage({
        type: 'gameState',
        workerId: workerId,
        gameState: {
          tRex: state.tRex,
          obstacles: state.obstacles,
          score: state.score,
          gameOver: state.gameOver,
        },
      });
    }

    // Enviar o estado final do jogo após o término
    const finalState = gameEngine.getState();
    parentPort.postMessage({
      type: 'gameState',
      workerId: workerId,
      gameState: {
        tRex: finalState.tRex,
        obstacles: finalState.obstacles,
        score: finalState.score,
        gameOver: finalState.gameOver,
      },
    });

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
