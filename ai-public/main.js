// main.js – Fluxo principal do GA com suporte a modos Treinamento e Teste

// Detecta o modo com base no nome do arquivo (se contém "testing", então é modo teste)
let testMode = window.location.pathname.includes("testing");
window.testMode = testMode; // define a flag global

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const infoDiv = document.getElementById('info');

// Elementos de controle (diferentes para treinamento e teste)
let startButton, stopButton, switchModeButton, popInput, eliteInput;
if (!testMode) {
  // Modo Treinamento
  startButton = document.getElementById('startTraining');
  stopButton = document.getElementById('stopTraining');
  switchModeButton = document.getElementById('switchToTest');
  popInput = document.getElementById('populationSize');
  eliteInput = document.getElementById('elitePercent');
} else {
  // Modo Teste
  startButton = document.getElementById('startTest');
  stopButton = document.getElementById('stopTest');
  switchModeButton = document.getElementById('switchToTraining');
}

let populationSize = (!testMode && popInput) ? parseInt(popInput.value) : 1; // no teste, apenas 1 dino
let elitePercent = (!testMode && eliteInput) ? parseFloat(eliteInput.value) : 0; // no teste, não usado

let generationCount = 1;
let engine;
let workers = [];
const numWorkers = (!testMode) ? (navigator.hardwareConcurrency || 4) : 0;
let dinosPerWorker = (!testMode) ? Math.ceil(populationSize / numWorkers) : 0;
let workersCompleted = 0;
let globalBestModel = null; // será carregado do servidor no modo teste
let globalBestFitness = 0;
let bestAllTime = 0;      // variavel para o best score de todos os tempos

let lastEliteSuccessRate = null;

// Função para carregar o melhor modelo do servidor
async function loadBestModelFromServer() {
  try {
    const response = await fetch('/api/best-model');
    if (!response.ok) {
      throw new Error('Nenhum modelo salvo encontrado');
    }
    const data = await response.json();
    console.log('Melhor modelo carregado do servidor:', data);
    return data;
  } catch (err) {
    console.error(err);
    return null;
  }
}

// Função para carregar o best score do servidor
async function loadBestScoreFromServer() {
  try {
    const response = await fetch('/api/best-score');
    if (!response.ok) throw new Error("Nenhum best score encontrado");
    const data = await response.json();
    console.log("Best score carregado:", data.bestScore);
    return data;
  } catch (err) {
    console.error(err);
    return null;
  }
}

// Loop de atualização para treinamento
function trainingGameLoop() {
  const ended = engine.updateAll();
  if (ended) {
    onAllDinosDead();
  }
  if (!testMode && workers.length > 0) {
    for (let i = 0; i < numWorkers; i++) {
      const startIdx = i * dinosPerWorker;
      const endIdx = startIdx + dinosPerWorker;
      const dinosSubset = engine.dinos.slice(startIdx, endIdx);
      workers[i].postMessage({
        dinos: dinosSubset,
        obstacles: engine.obstacles,
        generation: generationCount
      });
    }
  }
  renderGame();
  
  const aliveCount = engine.dinos.filter(d => d.alive).length;
  const totalCount = engine.dinos.length;
  const bestFitness = Math.max(...engine.dinos.map(d => d.fitness));
  const elite = window.selectElite(engine.dinos, elitePercent);
  const eliteAlive = elite.filter(d => d.alive).length;
  const eliteSuccessRate = (elite.length > 0) ? (eliteAlive / elite.length * 100) : 0;
  
  infoDiv.innerText = `Geração: ${generationCount} | Vivos: ${aliveCount}/${totalCount} | Melhor Fitness (Score): ${bestFitness.toFixed(2)} | Elite: ${eliteSuccessRate.toFixed(2)}%`;
  
  if (lastEliteSuccessRate === null || Math.abs(eliteSuccessRate - lastEliteSuccessRate) >= 5) {
    console.log(`Geração ${generationCount} - Elite survival rate: ${eliteSuccessRate.toFixed(2)}%`);
    lastEliteSuccessRate = eliteSuccessRate;
  }
  
  requestAnimationFrame(trainingGameLoop);
}

// Loop de atualização para teste
function testGameLoop() {
  engine.dinos.forEach(dino => {
    if (dino.alive) {
      dino.action = decideTestAction(dino, engine);
    }
  });
  const ended = engine.updateAll();
  if (ended) {
    engine.reset([new Dino(globalBestModel)]);
  }
  renderGame();
  
  const score = engine.dinos[0].fitness;
  // Atualiza o bestAllTime se o score atual ultrapassar o valor armazenado
  if (score > bestAllTime) {
    bestAllTime = score;
    // Salva o novo best score no servidor
    fetch('/api/save-best-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bestScore: bestAllTime })
    }).then(response => response.json())
      .then(data => console.log(data.message))
      .catch(err => console.error(err));
  }
  infoDiv.innerText = `Modo Teste | Score: ${score.toFixed(2)} | Melhor de Todos os Tempos: ${bestAllTime.toFixed(2)}`;
  
  requestAnimationFrame(testGameLoop);
}

// Renderização do jogo
function renderGame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'red';
  engine.obstacles.forEach(obs => {
    ctx.fillRect(
      obs.xPos,
      canvas.height - obs.height - obs.yPos,
      obs.width,
      obs.height
    );
  });
  engine.dinos.forEach((dino, idx) => {
    if (!dino.alive) return;
    ctx.fillStyle = `hsl(${(idx * 10) % 360}, 70%, 50%)`;
    ctx.fillRect(
      dino.x,
      canvas.height - dino.height - dino.y,
      dino.width,
      dino.height
    );
  });
}

// Função de decisão para teste (utilizando os pesos do cérebro)
function decideTestAction(dino, engine) {
  const state = engine.getStateForDino(dino);
  const weights = dino.brain.weights;
  if (weights.length < 15) return 2;
  let scores = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    let dot = 0;
    for (let i = 0; i < 5; i++) {
      dot += state[i] * weights[a * 5 + i];
    }
    scores[a] = dot;
  }
  return scores.indexOf(Math.max(...scores));
}

// Quando todos os dinos morrem (modo treinamento)
function onAllDinosDead() {
  const dinos = engine.dinos;
  const elite = window.selectElite(dinos, elitePercent);
  const best = elite[0];
  const bestFitness = best ? best.fitness : 0;
  
  console.log(`Geração ${generationCount} encerrada. Melhor Fitness local (da elite): ${bestFitness.toFixed(2)}`);
  
  // Se o melhor da elite desta geração for superior ao global, atualiza e salva
  if (bestFitness > globalBestFitness) {
    globalBestFitness = bestFitness;
    globalBestModel = best.brain;
    console.log(`Novo melhor global encontrado: Fitness = ${bestFitness.toFixed(2)}`);
    fetch('/api/save-best-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bestModel: globalBestModel, bestFitness })
    }).then(response => response.json())
      .then(data => console.log(data.message))
      .catch(err => console.error(err));
  } else {
    console.log(`Nenhuma melhoria: Fitness local ${bestFitness.toFixed(2)} não excede o global ${globalBestFitness.toFixed(2)}`);
  }
  
  const newPop = window.createNextGeneration(dinos, populationSize, elitePercent);
  generationCount++;
  engine.reset(newPop);
  lastEliteSuccessRate = null;
}



// Configura os workers para treinamento
function initWorkers() {
  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker('worker.js');
    worker.onmessage = function(e) {
      const { updatedDinos } = e.data;
      const startIdx = i * dinosPerWorker;
      for (let j = 0; j < updatedDinos.length; j++) {
        Object.assign(engine.dinos[startIdx + j], updatedDinos[j]);
      }
      workersCompleted++;
      if (workersCompleted === numWorkers) {
        workersCompleted = 0;
      }
    };
    worker.onerror = function(err) {
      console.error(`Worker[${i}] erro:`, err.message);
    };
    workers.push(worker);
  }
}

// Função de inicialização da engine – agora assíncrona para o modo de teste
async function initEngine() {
  if (!testMode) {
    populationSize = parseInt(popInput.value);
    elitePercent = parseFloat(eliteInput.value);
  } else {
    const bestData = await loadBestModelFromServer();
    if (bestData && bestData.bestModel) {
      globalBestModel = bestData.bestModel;
      console.log('Modelo carregado do servidor com fitness:', bestData.bestFitness);
    } else {
      alert("Nenhum modelo salvo encontrado. Execute o treinamento primeiro.");
      return;
    }
    const scoreData = await loadBestScoreFromServer();
    if (scoreData && scoreData.bestScore !== undefined) {
      bestAllTime = scoreData.bestScore;
    } else {
      bestAllTime = 0;
    }
  }
  const initialDinos = [];
  const count = testMode ? 1 : populationSize;
  for (let i = 0; i < count; i++) {
    if (testMode) {
      initialDinos.push(new Dino(globalBestModel));
    } else {
      initialDinos.push(new Dino());
    }
  }
  engine = new MultiDinoEngine(initialDinos.length);
  engine.dinos = initialDinos;
}

// Eventos dos botões de controle
if (startButton) {
  startButton.addEventListener('click', async () => {
    await initEngine();
    if (!testMode) {
      if (numWorkers > 0) initWorkers();
      requestAnimationFrame(trainingGameLoop);
      startButton.disabled = true;
      stopButton.disabled = false;
    } else {
      requestAnimationFrame(testGameLoop);
      startButton.disabled = true;
      stopButton.disabled = false;
    }
  });
}

if (stopButton) {
  stopButton.addEventListener('click', () => {
    window.location.reload();
  });
}

if (switchModeButton) {
  switchModeButton.addEventListener('click', () => {
    if (testMode) {
      window.location.href = 'training.html';
    } else {
      window.location.href = 'testing.html';
    }
  });
}
