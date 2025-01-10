/**************************************************
 * worker.js - Exemplo que usa MultiGameEngine
 **************************************************/
const { workerData, parentPort } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node-gpu');

// Carregue seu MultiGameEngine
const MultiGameEngine = require('./MultiGameEngine');

// -------------------------------------------------
// Dados recebidos do servidor (via workerData)
const workerId         = workerData.workerId;
const numDinos         = workerData.numDinos || 5;
const modelWeights     = workerData.modelWeights || [];
const currentGeneration= workerData.currentGeneration;
const maxGenerations   = workerData.maxGenerations || 100;

// -------------------------------------------------
// Config de simulação
const MAX_STEPS_PER_RUN = 3000;  // ex: limite de steps pra não travar
let explorationRate = 0.4;       // Exemplo de "exploração" (fica a seu critério)
const explorationDecay = 0.02;   // Decaimento a cada geração, etc.

// -------------------------------------------------
// Variável global: Modelo
let model = null;

// -------------------------------------------------
// Cria o modelo (mesma arquitetura do server)
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 24,
    activation: 'relu',
    inputShape: [7],
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  model.add(tf.layers.dense({
    units: 3, // 3 ações
    activation: 'softmax',
    useBias: true
  }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

// Carrega pesos no modelo
async function loadModel() {
  model = createModel();
  if (modelWeights.length > 0) {
    try {
      const weightTensors = modelWeights.map((w) =>
        tf.tensor(w, undefined, 'float32')
      );
      model.setWeights(weightTensors);
      console.log(`[Worker ${workerId}] Pesos do modelo carregados.`);
    } catch (err) {
      console.error(`[Worker ${workerId}] Erro ao setar pesos:`, err);
      process.exit(1);
    }
  } else {
    console.log(`[Worker ${workerId}] Modelo sem pesos (aleatórios).`);
  }
}

// -------------------------------------------------
// Normalização simples do estado de cada Dino
function normalizeState(dinoState) {
  // dinoState = { tRexY, velocityY, obstacleX, obstacleW, obstacleH, alive }
  // Ajuste escalas conforme seu jogo
  const maxDistance = 600;
  const maxTReXJumpHeight = 120;
  const maxVelocityY = 20;
  const maxObstacleWidth = 60;
  const maxObstacleHeight = 60;

  const yNormalized       = (dinoState.tRexY / maxTReXJumpHeight);
  const vYNormalized      = (dinoState.velocityY / maxVelocityY); // pode ficar -1 ~ +1
  const distanceToObstacle= (dinoState.obstacleX !== null)
                            ? dinoState.obstacleX / maxDistance
                            : 1; 
  const obsW = (dinoState.obstacleW / maxObstacleWidth);
  const obsH = (dinoState.obstacleH / maxObstacleHeight);
  const obstacleInSight = (dinoState.obstacleX !== null) ? 1 : 0;
  const onGround         = (dinoState.tRexY === 0) ? 1 : 0;

  // Monta array final
  // Ex: [y, vY, dist, obsW, obsH, obstacleInSight, onGround]
  // Ajuste se precisar
  const arr = [
    Math.max(Math.min(yNormalized, 1), 0),
    Math.max(Math.min(vYNormalized,  1), -1),
    Math.max(Math.min(distanceToObstacle, 1), 0),
    Math.max(Math.min(obsW, 1), 0),
    Math.max(Math.min(obsH, 1), 0),
    obstacleInSight,
    onGround
  ];
  return arr;
}

// -------------------------------------------------
// Escolher ação para cada Dino
function chooseActionsForAllDinos(dinoStates) {
  // Retornamos um array de tamanho = dinoStates.length
  // com a Ação de cada Dino
  const actions = [];

  tf.tidy(() => {
    // Montar batch de states
    // Filtrar: se dino morto, podemos "0" ou "2" ou algo fixo...
    const inputData = [];
    for (let st of dinoStates) {
      if (!st.alive) {
        // Dino morto => action = 2 (nada), por ex
        inputData.push([0, 0, 0, 0, 0, 0, 0]);
      } else {
        inputData.push(normalizeState(st));
      }
    }

    const inputTensor = tf.tensor2d(inputData, [dinoStates.length, 7], 'float32');
    const predictProbs = model.predict(inputTensor); // shape [N,3]

    const probsArray = predictProbs.arraySync(); 
    for (let i = 0; i < dinoStates.length; i++) {
      if (!dinoStates[i].alive) {
        actions.push(2);
        continue;
      }

      // Exploração
      if (Math.random() < explorationRate) {
        // random
        const randomAct = Math.floor(Math.random() * 3);
        actions.push(randomAct);
      } else {
        // argMax
        const arrP = probsArray[i];
        let bestAct = 0;
        let bestVal = -999;
        arrP.forEach((val, idx) => {
          if (val > bestVal) {
            bestVal = val;
            bestAct = idx;
          }
        });
        actions.push(bestAct);
      }
    }
  });

  return actions;
}

// -------------------------------------------------
// Roda a simulação de 1 "episódio"
function runSimulation(numDinos) {
  // Cria engine
  const engine = new MultiGameEngine(numDinos);

  let stepCount = 0;
  while (!engine.allDinosDead && stepCount < MAX_STEPS_PER_RUN) {
    stepCount++;

    // Pega states para cada Dino
    const states = engine.getStatesForAllDinos();
    // Decide ação para cada Dino
    const actions = chooseActionsForAllDinos(states);
    // update no engine
    engine.updateAllDinos(actions);
  }

  // Ao terminar (todos morreram ou steps excedeu)
  // Extrair fitness => ex: média dos scores?
  let sumScore = 0;
  let maxScore = 0;
  engine.dinos.forEach(d => {
    sumScore += d.score;
    if (d.score > maxScore) maxScore = d.score;
  });

  // Você pode escolher: "média", "soma", "máximo", etc.
  // Exemplo: pego a média
  const avgScore = sumScore / engine.dinos.length;
  return avgScore;  // fitness
}

// -------------------------------------------------
// Função principal do worker
(async () => {
  await loadModel();

  // Rodamos 1 simulação (ou várias, se quiser)
  const fitness = runSimulation(numDinos);

  // Faz decaimento da exploração (opcional)
  if (explorationRate > 0.01) {
    explorationRate -= explorationDecay;
    if (explorationRate < 0.01) explorationRate = 0.01;
  }

  console.log(`[Worker ${workerId}] Fitness = ${fitness.toFixed(2)}`);

  // Retorna para o parent
  parentPort.postMessage({
    type: 'fitness',
    workerId,
    fitness
  });

  // Worker encerra
  process.exit(0);
})().catch(err => {
  console.error(`[Worker ${workerId}] ERRO:`, err);
  process.exit(1);
});
