// test.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const puppeteer = require('puppeteer');
const fs = require('fs');

// ------------------------------------------------------
// Ajuste aqui se quiser outro workerNumber fixo, mas
// neste caso, estamos focando no "worker 49".
// Você pode alterar para outro valor se necessário.
// ------------------------------------------------------
const WORKER_NUMBER = 29;

// Função para retornar um array de pastas que seguem o padrão 'model-worker-49-generation-XXX'
// e extrair o número de generation
function findWorker49Models(modelsRootDir) {
  // Ler tudo que tem na pasta 'models'
  const items = fs.readdirSync(modelsRootDir, { withFileTypes: true });
  
  // Filtrar somente diretórios no formato "model-worker-49-generation-XYZ"
  const matchedFolders = [];
  for (const dirent of items) {
    if (dirent.isDirectory()) {
      const folderName = dirent.name;
      const pattern = new RegExp(`^model-worker-${WORKER_NUMBER}-generation-(\\d+)$`);
      const match = folderName.match(pattern);
      if (match) {
        // Capturar a generation
        const generationNum = parseInt(match[1], 10);
        matchedFolders.push({
          folderName,
          generation: generationNum
        });
      }
    }
  }

  // Ordenar pelo número da generation de forma ascendente
  matchedFolders.sort((a, b) => a.generation - b.generation);
  return matchedFolders;
}

// Carregar modelo do disco
async function loadModel(folderPath) {
  try {
    // Montar caminho absoluto do model.json
    const modelJsonPath = path.join(folderPath, 'model.json'); 
    if (!fs.existsSync(modelJsonPath)) {
      console.log(`Arquivo model.json não encontrado em ${modelJsonPath}. Pulando...`);
      return null;
    }

    // Exemplo: "file://C:/Users/.../model-worker-49-generation-88/model.json"
    const localPath = path.resolve(modelJsonPath);
    const fileUrlPath = 'file://' + localPath.replace(/\\/g, '/');

    console.log(`Carregando modelo '${path.basename(folderPath)}' de: ${fileUrlPath}`);
    const model = await tf.loadLayersModel(fileUrlPath);
    console.log(`Modelo '${path.basename(folderPath)}' carregado com sucesso.`);
    return model;
  } catch (error) {
    console.error(`Falha ao carregar o modelo '${path.basename(folderPath)}':`, error);
    return null;
  }
}

// Normalização do estado do jogo
function normalizeState(state) {
  const maxDistance = 600;       // Distância máxima para obstáculos
  const maxTReXVY = 20;          // Velocidade vertical máxima do T-Rex
  const maxTReXJumpHeight = 100; // Altura máxima de salto do T-Rex
  const maxObstacleWidth = 50;   // Largura máxima do obstáculo
  const maxObstacleHeight = 50;  // Altura máxima do obstáculo

  const distanceToObstacle = state.obstacleX !== null
    ? (state.obstacleX - state.tRexX) / maxDistance
    : 1;

  const obstacleInSight = state.obstacleX !== null ? 1 : 0;
  const tRexOnGround = state.tRexY === 0 ? 1 : 0;

  return [
    (state.tRexY + maxTReXJumpHeight) / maxTReXJumpHeight,
    state.tRexVY / maxTReXVY,
    distanceToObstacle,
    (state.obstacleWidth  || 0) / maxObstacleWidth,
    (state.obstacleHeight || 0) / maxObstacleHeight,
    obstacleInSight,
    tRexOnGround
  ];
}

// Escolher ação com base no modelo
function chooseAction(normalizedState, model, explorationRate = 0.05) {
  return tf.tidy(() => {
    const stateTensor = tf.tensor2d([normalizedState], undefined, 'float32');
    const actionProbs = model.predict(stateTensor);

    // Simplesmente escolher a ação de maior probabilidade (poderia ter uma taxa de exploração)
    if (Math.random() < explorationRate) {
      return Math.floor(Math.random() * 3); // 0, 1 ou 2
    }

    const action = actionProbs.argMax(1).dataSync()[0];
    return action;
  });
}

// Mapear ação numérica para comando do jogo
function mapActionToCommand(action) {
  switch(action) {
    case 0: return 'jump';
    case 1: return 'duck';
    case 2: return 'do_nothing';
    default: return 'do_nothing';
  }
}

// Iniciar um servidor + abrir a tela do Dino para cada modelo
function startServerForModel(folderName, generation, model, index) {
  const app = express();
  app.use(express.static(path.join(__dirname, 'public-test')));
  const server = http.createServer(app);
  const io = socketIo(server);

  io.on('connection', (socket) => {
    console.log(`Cliente conectado para o modelo '${folderName}', generation=${generation}`);
    socket.emit('startGame');

    socket.on('state', (gameState) => {
      const state = {
        tRexY: gameState.dinoY,
        tRexVY: gameState.dinoVY,
        tRexX: gameState.dinoX,
        obstacleX: gameState.obstacleX,
        obstacleWidth: gameState.obstacleWidth,
        obstacleHeight: gameState.obstacleHeight,
        currentSpeed: gameState.currentSpeed,
      };

      if (state.obstacleX === null || state.obstacleX === undefined) {
        state.obstacleX = 600;
      }

      const normalizedState = normalizeState(state);
      const actionIndex = chooseAction(normalizedState, model);
      const actionCommand = mapActionToCommand(actionIndex);
      socket.emit('action', actionCommand);
    });

    socket.on('disconnect', () => {
      console.log(`Cliente desconectado do modelo '${folderName}', generation=${generation}`);
    });
  });

  // Base da porta 3002 para não conflitar com outro
  const port = 3002 + index; 
  server.listen(port, () => {
    console.log(`Servidor do modelo '${folderName}' rodando na porta ${port}`);
  });

  openTestDinoGame(port, folderName, generation);
}

async function openTestDinoGame(port, folderName, generation) {
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
  const url = `http://localhost:${port}`;
  await page.goto(url);

  // Desabilitar ocultar
  await page.evaluateOnNewDocument(() => {
    Object.defineProperty(document, 'hidden', { value: false });
    Object.defineProperty(document, 'visibilityState', { value: 'visible' });
    document.addEventListener(
      'visibilitychange',
      (event) => {
        event.stopImmediatePropagation();
      },
      true
    );
  });

  console.log(`Abrindo o Dino Game p/ modelo '${folderName}' (gen=${generation}) em ${url}`);
}

// ----- FUNÇÃO PRINCIPAL -----
async function init() {
  const rootModelsDir = path.resolve(__dirname, 'models');

  // 1) Procurar todos os diretórios do tipo "model-worker-49-generation-xxx"
  let items = fs.readdirSync(rootModelsDir, { withFileTypes: true });
  const worker49Models = [];

  for (const dirent of items) {
    if (dirent.isDirectory()) {
      const folderName = dirent.name; 
      // Padrão "model-worker-49-generation-XYZ"
      const match = folderName.match(/^model-worker-49-generation-(\d+)$/);
      if (match) {
        const generation = parseInt(match[1], 10);
        worker49Models.push({ folderName, generation });
      }
    }
  }

  // 2) Ordenar em ordem ascendente de generation
  worker49Models.sort((a, b) => a.generation - b.generation);

  if (worker49Models.length === 0) {
    console.log('Não há nenhum modelo para "worker-49". Encerrando...');
    return;
  }

  // 3) Selecionar somente os últimos 20
  const last20 = worker49Models.slice(-20);
  console.log(`Encontrados ${worker49Models.length} modelos com 'worker-49'. Abrindo os últimos 20.`);
  console.log(last20.map(item => item.folderName));

  // 4) Para cada um dos últimos 20, carregar o modelo e abrir o servidor
  for (let i = 0; i < last20.length; i++) {
    const { folderName, generation } = last20[i];
    const folderPath = path.join(rootModelsDir, folderName);
    
    // Carregar modelo
    const model = await loadModel(folderPath);
    if (!model) {
      console.log(`Não foi possível carregar o modelo: ${folderName}. Pulando...`);
      continue;
    }
    
    // Subir servidor e abrir aba
    startServerForModel(folderName, generation, model, i);
  }
}

init();
