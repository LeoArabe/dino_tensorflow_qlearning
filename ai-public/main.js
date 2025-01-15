/**************************************************
 * main.js — Fluxo Principal (versão unificada)
 **************************************************/

// Referências ao canvas e contexto
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Parâmetros de pop/tamanho/controle
let populationSize = 20000;
let displayEliteSize = 1000;
let generationCount = 1;
let engine = null;

// Referências DOM
const chkShowElite = document.getElementById('chkShowElite');
const infoDiv = document.getElementById('info');

// Configurações de Workers
const numWorkers = navigator.hardwareConcurrency || 8;
const workers = [];
for (let i = 0; i < numWorkers; i++) {
  workers.push(new Worker('worker.js'));
}

// Quantos dinos por Worker
let dinosPerWorker = Math.ceil(populationSize / numWorkers);
let workersCompleted = 0;

// Controle de "best" da geração
let generationBestFitness = 0;
let generationBestModel = null;

// Controle de "best" global
let globalBestFitness = 0;
let globalBestModel = null;

// Flag para controlar se podemos disparar updates
let canUpdate = true;

// Controle de FPS (para não sobrecarregar)
let lastRenderTime = 0;
const desiredFPS = 30; 
const frameInterval = 1000 / desiredFPS;

// ────────────────────────────────────────────────────────────
// 1) Receber dados dos Workers
// ────────────────────────────────────────────────────────────
workers.forEach((worker, wIndex) => {
  worker.onmessage = (e) => {
    const { updatedDinos, generation, bestFitness, bestModel } = e.data;

    // Aplicar as mudanças nos dinos correspondentes a este Worker
    const start = wIndex * dinosPerWorker;
    for (let i = 0; i < updatedDinos.length; i++) {
      Object.assign(engine.dinos[start + i], updatedDinos[i]);
    }

    // Atualizar "melhor" dentro desta rodada
    if (bestFitness > generationBestFitness) {
      generationBestFitness = bestFitness;
      generationBestModel = bestModel;
    }

    // Verifica se todos os Workers finalizaram
    workersCompleted++;
    if (workersCompleted === numWorkers) {
      // Reset para próxima
      workersCompleted = 0;

      // Se todos morreram
      if (engine.dinos.every(d => !d.alive)) {
        onAllDinosDead(); // Finalizamos geração aqui
      }

      // Obs.: REMOVIDO o salvamento aqui para não ficar gravando toda hora
      //       (ficava chamando saveBestModel() a cada "update" quando batia recorde)

      // Mostrar info no DOM
      infoDiv.innerText = `Geração: ${generationCount} | BestGlobal: ${globalBestFitness.toFixed(2)}`;

      // Render
      renderGame();

      // Liberar para próximo update
      canUpdate = true;
    }
  };

  // Em caso de erro no Worker
  worker.onerror = (err) => {
    console.error(`Worker[${wIndex}] erro:`, err.message);
  };
});

// ────────────────────────────────────────────────────────────
// 2) Funções principais de controle do GA e do engine
// ────────────────────────────────────────────────────────────

// Inicializa a engine e começa uma geração
function startGeneration(newDinos = null) {
  // Se newDinos está definido, iremos usar esse array
  // senão, criamos com 'populationSize'
  engine = new MultiDinoEngine(newDinos ? newDinos.length : populationSize);
  if (newDinos) {
    engine.dinos = newDinos;
  }
  console.log(`-> Geração ${generationCount} iniciada (pop=${engine.dinos.length}).`);
}

// Chamado quando TODOS os dinos morreram
function onAllDinosDead() {
  const dinos = engine.dinos;
  const elite = selectElite(dinos, Math.floor(populationSize * 0.05));
  const best = elite[0];
  const bestFitness = best ? best.fitness : 0;

  console.log(`Geração #${generationCount} terminou. Melhor Fitness local: ${bestFitness.toFixed(2)}`);

  // Se o melhor da geração é maior que o global => atualiza e salva
  if (bestFitness > globalBestFitness) {
    globalBestFitness = bestFitness;
    globalBestModel = best.brain;
    console.log(`>>> Novo melhor global encontrado: fitness = ${globalBestFitness.toFixed(2)}`);

    // Salva no back-end
    saveBestModel(globalBestModel, globalBestFitness, true);
  }

  // Cria nova pop e reseta engine
  const newPop = createNextGeneration(dinos, populationSize);
  generationCount++;
  engine.reset(newPop);

  console.log(`-> Nova Geração #${generationCount} iniciada.`);

  // Reset as variáveis de best da geração
  generationBestFitness = 0;
  generationBestModel = null;
}

// Dispara atualização do jogo via Workers
function updateGameState() {
  workers.forEach((worker, wIndex) => {
    const start = wIndex * dinosPerWorker;
    const end = start + dinosPerWorker;
    const dinosSubset = engine.dinos.slice(start, end);

    // Mandamos para o Worker
    worker.postMessage({
      dinos: dinosSubset,
      obstacles: engine.obstacles,
      generation: generationCount
    });
  });
}

// Desenha no canvas
function renderGame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Desenhar obstáculos
  ctx.fillStyle = 'red';
  engine.obstacles.forEach(obs => {
    ctx.fillRect(
      obs.xPos,
      canvas.height - obs.height - obs.yPos,
      obs.width,
      obs.height
    );
  });

  // Quais dinos desenhar (se "Mostrar Elite" estiver marcado, filtra)
  let dinosToRender = chkShowElite.checked
    ? selectElite(engine.dinos, displayEliteSize)
    : engine.dinos;

  // Desenhar dinos
  dinosToRender.forEach((d, idx) => {
    if (!d.alive) return;
    ctx.fillStyle = `hsl(${(idx * 10) % 360}, 70%, 50%)`;
    ctx.fillRect(
      d.x,
      canvas.height - d.height - d.y,
      d.width,
      d.height
    );
  });
}

// ────────────────────────────────────────────────────────────
// 3) Loop principal com controle de FPS
// ────────────────────────────────────────────────────────────
function gameLoop(timestamp) {
  if (!lastRenderTime) lastRenderTime = timestamp;
  const delta = timestamp - lastRenderTime;

  if (delta >= frameInterval && canUpdate) {
    lastRenderTime = timestamp - (delta % frameInterval);
    canUpdate = false;

    // 1) Atualiza local (mov. obstáculos, check se morreram)
    const ended = engine.updateAll(); 
    if (ended) {
      onAllDinosDead();
    }

    // 2) Chama Workers para definir ações
    updateGameState();
  }

  requestAnimationFrame(gameLoop);
}

// ────────────────────────────────────────────────────────────
// 4) Salvamento de modelo no servidor
// ────────────────────────────────────────────────────────────
function saveBestModel(model, fitness, isBestAllTime = false) {
  const modelData = JSON.stringify(model); // Rede neural (brain)
  fetch('/save-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelData,
      fitness,
      bestAllTime: isBestAllTime
    })
  })
    .then(response => response.json())
    .then(data => {
      console.log(`Modelo salvo: ${data.filename}`);
    })
    .catch(error => {
      console.error('Erro ao salvar o modelo:', error);
    });
}

// ────────────────────────────────────────────────────────────
// 5) Inicialização
// ────────────────────────────────────────────────────────────
startGeneration();
requestAnimationFrame(gameLoop);
