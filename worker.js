/************************************************** 
 * worker.js
 **************************************************/
const { workerData, parentPort } = require('worker_threads');
const GameEngine = require('./gameEngine');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');

// -------------------------------------------------
// Dados que chegam do servidor (workerData)
const workerId = workerData.workerId;
const numInstances = workerData.numInstances || 1;
const modelWeights = workerData.modelWeights || [];
const currentGeneration = workerData.currentGeneration;
const maxGenerations = workerData.maxGenerations;

// -------------------------------------------------
// Engine(s) de jogo
const gameEngines = [];
for (let i = 0; i < numInstances; i++) {
  gameEngines.push(new GameEngine());
}

// -------------------------------------------------
// Configurações de Exploração, Forçar Pulo, etc.

// EXPLORAÇÃO
let explorationRate = 1.0;            // Inicia com 100% de exploração
const minExplorationRate = 0.1;       // Taxa mínima de exploração
// Diminui bem mais devagar (ex: -0.001 por geração)
const explorationDecay = 0.001;       

// AJUSTE DO dynamicMinFitness
let dynamicMinFitness = 15;           // Fitness mínimo inicial bem menor
// Em vez de 10%, cresce 1% a cada final de geração
const fitnessGrowthRate = 0.01;       // 1% de crescimento

// Exemplo simples de probabilidade de forçar pulo 
// (Queremos que logo de cara o Dino explore + jumping).
function getForceJumpProbability(gen) {
  // Ex: se gen < 10 => p = 0.6, se gen < 20 => p = 0.3, depois 0
  if (gen < 10) {
    return 0.6;
  } else if (gen < 20) {
    return 0.3;
  } else {
    return 0.0;
  }
}

// -------------------------------------------------
// Pasta e arquivo de LOG
const logsDir = path.resolve(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}
const logFilePath = path.join(logsDir, `worker_${workerId}_log.txt`);

// Função auxiliar: registra mensagens no log
function logMessage(msg, level = 'DEBUG') {
  const time = new Date().toISOString();
  const line = `[${time}] [Worker ${workerId}] [${level}] ${msg}\n`;
  fs.appendFileSync(logFilePath, line, 'utf8');
  // Também exibe no console:
  console.log(line.trim());
}

// -------------------------------------------------
// Função para garantir pasta "models"
function ensureModelsDirectory() {
  const modelsDir = path.resolve(__dirname, 'models');
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
    logMessage(`Pasta "models" criada com sucesso.`, 'INFO');
  } else {
    logMessage(`Pasta "models" já existe.`, 'INFO');
  }
}

// -------------------------------------------------
// Cria o modelo
function createModel() {
  const model = tf.sequential();

  // Camada de Entrada
  model.add(tf.layers.dense({
    units: 24,
    activation: 'relu',
    inputShape: [7], // 7 entradas normalizadas
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  // Camada Oculta 1
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  // Camada de Dropout
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Camada Oculta 2
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  // Camada de Saída (3 ações)
  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax',
    useBias: true
  }));

  return model;
}

// -------------------------------------------------
// Variáveis globais
let model;
let bestFitness = -Infinity;  
// Nome do arquivo do "melhor modelo" (para sobrescrever e não acumular)
const bestModelFilename = `model-best-worker-${workerId}`;

// -------------------------------------------------
// Carrega ou inicializa o modelo com os pesos
async function loadModel() {
  model = createModel();

  if (modelWeights.length > 0) {
    try {
      const weightTensors = modelWeights.map((w, index) => {
        const tensor = tf.tensor(w, undefined, 'float32');
        logMessage(`Peso ${index} shape: ${tensor.shape}`, 'DEBUG');
        return tensor;
      });
      model.setWeights(weightTensors);
      logMessage(`Pesos carregados com sucesso.`, 'INFO');
    } catch (error) {
      logMessage(`Erro ao carregar pesos - ${error}`, 'ERROR');
      process.exit(1);
    }
  } else {
    logMessage(`Inicializando modelo com pesos aleatórios.`, 'INFO');
  }

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  logMessage(`Modelo compilado com sucesso.`, 'INFO');
}

// -------------------------------------------------
// Normalizar estado do jogo
function normalizeState(state) {
  const maxDistance = 600;
  const maxTRexJumpHeight = 100;
  const maxVelocityY = 20;
  const maxObstacleWidth = 50;
  const maxObstacleHeight = 50;

  let normalizedTRexY = state.tRexY / maxTRexJumpHeight;
  normalizedTRexY = Math.max(Math.min(normalizedTRexY, 1), 0);

  let normalizedTRexVY = state.tRexVelocityY / maxVelocityY;
  // Aqui, deixei com range -1 a +1:
  normalizedTRexVY = Math.max(Math.min(normalizedTRexVY, 1), -1);

  const distanceToObstacle = state.obstacleX !== null
    ? (state.obstacleX - state.tRexX) / maxDistance
    : 1;
  const normalizedDistance = Math.max(Math.min(distanceToObstacle, 1), 0);

  const obstacleInSight = state.obstacleX !== null ? 1 : 0;
  const tRexOnGround = state.tRexY === 0 ? 1 : 0;

  const normalizedObstacleWidth = (state.obstacleWidth || 0) / maxObstacleWidth;
  const normalizedObstacleHeight = (state.obstacleHeight || 0) / maxObstacleHeight;

  const normalizedState = [
    normalizedTRexY,
    normalizedTRexVY,
    normalizedDistance,
    normalizedObstacleWidth,
    normalizedObstacleHeight,
    obstacleInSight,
    tRexOnGround
  ];

  // Verificar e substituir NaNs
  normalizedState.forEach((value, index) => {
    if (isNaN(value)) {
      logMessage(`Valor NaN no índice ${index} do estado normalizado. Substituindo por 0`, 'DEBUG');
      normalizedState[index] = 0;
    }
  });

  logMessage(`Estado Normalizado: [${normalizedState.map(v => v.toFixed(3)).join(', ')}]`, 'DEBUG');
  return normalizedState;
}

// -------------------------------------------------
// Decay da taxa de exploração
function decayExplorationRate() {
  if (explorationRate > minExplorationRate) {
    explorationRate -= explorationDecay;
    explorationRate = Math.max(explorationRate, minExplorationRate);
    logMessage(`Taxa de Exploração atualizada: ${explorationRate.toFixed(3)}`, 'INFO');
  } else {
    logMessage(`Taxa de Exploração atingiu o mínimo: ${minExplorationRate}`, 'INFO');
  }
}

// -------------------------------------------------
// Função para escolher ação (com forçar pulo e etc.)
function chooseAction(state) {
  return tf.tidy(() => {
    // 1) Forçar pulo com base na geração e prob.
    const forceJumpProb = getForceJumpProbability(currentGeneration);

    if (Math.random() < forceJumpProb) {
      logMessage(`Forçando pulo (Gen=${currentGeneration}, prob=${forceJumpProb})`, 'DEBUG');
      return 0; // 0 => pular
    }

    // 2) Usa o modelo + explor.
    const normalizedState = normalizeState(state);
    const stateTensor = tf.tensor2d([normalizedState], undefined, 'float32');
    const actionProbs = model.predict(stateTensor);

    if (!actionProbs) {
      logMessage(`actionProbs é undefined. Retornando '0' (pular) como fallback`, 'DEBUG');
      return 0;
    }

    const probs = actionProbs.dataSync();
    const sumProbs = probs.reduce((a, b) => a + b, 0);
    logMessage(`Prob. Modelo=[${probs.map(v => v.toFixed(3)).join(', ')}], soma=${sumProbs.toFixed(3)}`, 'DEBUG');

    // Exploração
    if (Math.random() < explorationRate) {
      const randomAction = Math.floor(Math.random() * 3);
      logMessage(`Ação random escolhida: ${randomAction}`, 'DEBUG');
      return randomAction;
    }

    // Usa previsões
    const bestAction = tf.argMax(actionProbs, 1).dataSync()[0];
    logMessage(`Ação (modelo) escolhida: ${bestAction}`, 'DEBUG');
    return bestAction;
  });
}

// -------------------------------------------------
// Função para salvar (SE realmente for melhor)
async function saveModelIfBetter(currentFitness, generationNumber) {
  // Critério: fitness > bestFitness E >= dynamicMinFitness
  if (currentFitness > bestFitness && currentFitness >= dynamicMinFitness) {
    bestFitness = currentFitness;
    const modelsDir = path.resolve(__dirname, 'models');
    // Em vez de criar um arquivo novo, sobrescrevo sempre o MESMO:
    const modelName = bestModelFilename; 
    const savePath = path.join(modelsDir, modelName);

    try {
      // Sobrescreve o mesmo model-better
      await model.save(`file://${savePath}`);
      logMessage(
        `Modelo SALVO (Sobrescrevendo) na geração ${generationNumber} com fitness ${currentFitness}. Caminho=${savePath}`,
        'INFO'
      );

      // Envia mensagem
      parentPort.postMessage({
        type: 'modelSaved',
        workerId,
        generation: generationNumber,
        fitness: currentFitness,
        path: savePath
      });
    } catch (error) {
      logMessage(`ERRO ao salvar modelo (Gen=${generationNumber}): ${error}`, 'ERROR');
    }
  } else {
    logMessage(
      `Fitness ${currentFitness} < bestFitness(${bestFitness}) ou < dynamicMinFitness(${dynamicMinFitness}). Não salva.`,
      'DEBUG'
    );
  }
}

// -------------------------------------------------
// Roda os jogos e calcula fitness
async function runGames(generationNumber) {
  let totalScore = 0;
  const numGames = gameEngines.length;

  for (let i = 0; i < numGames; i++) {
    const engine = gameEngines[i];
    // Reinicia o jogo
    engine.reset();

    logMessage(`[GEN=${generationNumber}] [GAME=${i+1}] Start`, 'INFO');

    while (!engine.gameOver) {
      const state = engine.getState();
      const action = chooseAction({
        tRexY: state.tRexY,
        tRexVelocityY: state.tRexVelocityY,
        tRexX: state.tRexX,
        obstacleX: state.obstacleX,
        obstacleWidth: state.obstacleWidth,
        obstacleHeight: state.obstacleHeight,
      });

      engine.update(action);
    }

    logMessage(`[GEN=${generationNumber}] [GAME=${i+1}] GameOver Score=${engine.score}`, 'INFO');
    totalScore += engine.score;
  }

  const averageScore = totalScore / numGames;
  logMessage(`Fitness médio (Gen=${generationNumber}, Worker=${workerId}) = ${averageScore}`, 'INFO');

  // Tenta salvar se melhor
  await saveModelIfBetter(averageScore, generationNumber);

  // Decair exploration
  decayExplorationRate();
}

// -------------------------------------------------
// Roda a geração
async function runGeneration(generationNumber) {
  logMessage(`Iniciando geração ${generationNumber}`, 'INFO');
  await runGames(generationNumber);

  // Crescimento do dynamicMinFitness em 1%
  dynamicMinFitness *= (1 + fitnessGrowthRate);
  logMessage(`Novo dynamicMinFitness = ${dynamicMinFitness.toFixed(2)}`, 'INFO');

  // Se ainda não chegou no máximo
  if (generationNumber < maxGenerations) {
    await runGeneration(generationNumber + 1);
  } else {
    logMessage(
      `Geração máxima (${maxGenerations}) atingida (Worker ${workerId}).`,
      'INFO'
    );
    parentPort.postMessage({
      type: 'done',
      workerId,
    });
  }
}

// -------------------------------------------------
// Inicialização do Worker
async function init() {
  ensureModelsDirectory();
  await loadModel();
  await runGeneration(currentGeneration + 1);
}

init().catch(error => {
  logMessage(`ERRO na inicialização Worker ${workerId}: ${error}`, 'ERROR');
  process.exit(1);
});
