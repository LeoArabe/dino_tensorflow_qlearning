/**************************************************
 * server.js
 **************************************************/
const express = require('express');
const http = require('http');
const socketIo = require('socket.io'); // se quiser usar sockets
const path = require('path');
const fs = require('fs');
const { Worker } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node-gpu');  // ou tfjs-node-gpu
const open = require('open');                // para abrir o browser

// -------------------------------------------------
// PARÂMETROS DO GA / TREINO
// -------------------------------------------------
const populationSize = 50;
const maxGenerations = 100;
const eliteSize = Math.floor(populationSize * 0.1);

let population = [];         // array de { model: tf.LayersModel, fitness: number }
let generationData = [];     // dados de cada geração para exibir no front
let overallBestFitness = -Infinity;
let overallBestModel = null;

// Pasta para salvar modelos
const modelsDir = path.join(__dirname, 'models');

// -------------------------------------------------
// EXPRESS / SERVER
// -------------------------------------------------
const app = express();
const server = http.createServer(app);
const io = socketIo(server); // Se quiser usar socket.io

// Rota principal que serve a página do front-end (engineTraining.html)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '', 'TrainingInterface.html'));
});

// Iniciar servidor
const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Servidor rodando em http://localhost:${PORT}`);
  // Abre automaticamente no navegador
  open(`http://localhost:${PORT}`);
});

// -------------------------------------------------
// GARANTIR PASTA "models"
function ensureModelsDirectory() {
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
    console.log('Pasta "models" criada com sucesso.');
  } else {
    console.log('Pasta "models" já existe.');
  }
}
ensureModelsDirectory();

// -------------------------------------------------
// CRIAÇÃO / INICIALIZAÇÃO DO MODELO
// -------------------------------------------------
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 24,
    activation: 'relu',
    inputShape: [7], // 7 entradas normalizadas
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  // Camada de Dropout
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    useBias: true
  }));

  model.add(tf.layers.dense({
    units: 3, // 0,1,2
    activation: 'softmax',
    useBias: true
  }));

  return model;
}

// -------------------------------------------------
// INICIALIZAR POPULAÇÃO
// -------------------------------------------------
function initializePopulation() {
  population = [];
  for (let i = 0; i < populationSize; i++) {
    const model = createModel();
    population.push({ model, fitness: null });
  }
}

// -------------------------------------------------
// GA: SELEÇÃO, CROSSOVER, MUTATION ...
// (exemplo simplificado — ajuste conforme sua lógica)
// -------------------------------------------------
function selectElite(pop) {
  return pop
    .sort((a, b) => b.fitness - a.fitness)
    .slice(0, eliteSize);
}

function tournamentSelection(pop) {
  // Exemplo fixo: torneio de 3
  const tSize = 3;
  const tournament = [];
  for (let i = 0; i < tSize; i++) {
    const r = Math.floor(Math.random() * pop.length);
    tournament.push(pop[r]);
  }
  tournament.sort((a, b) => b.fitness - a.fitness);
  return tournament[0]; 
}

function crossover(modelA, modelB) {
  // Exemplo simples: combina pesos 50/50
  const child = createModel();
  const wA = modelA.getWeights();
  const wB = modelB.getWeights();

  const newWeights = wA.map((tensorA, i) => {
    const tensorB = wB[i];
    // shape e dtype
    const shape = tensorA.shape;
    const dtype = tensorA.dtype;

    const arrA = tensorA.dataSync();
    const arrB = tensorB.dataSync();
    const newArr = arrA.map((val, idx) => {
      return Math.random() < 0.5 ? val : arrB[idx];
    });
    return tf.tensor(newArr, shape, dtype);
  });

  child.setWeights(newWeights);
  return child;
}

function mutate(model) {
  // Exemplo: chance de 1% para cada peso
  const rate = 0.01;
  const weights = model.getWeights();
  const mutated = weights.map(t => {
    const shape = t.shape;
    const dtype = t.dtype;
    const arr = t.dataSync();
    const newArr = arr.map((val) => {
      return Math.random() < rate
        ? val + (Math.random() * 0.2 - 0.1)
        : val;
    });
    return tf.tensor(newArr, shape, dtype);
  });
  model.setWeights(mutated);
}

function createNextGeneration(elite) {
  const newPop = [];

  // mantém a elite
  elite.forEach(e => {
    newPop.push({ model: e.model, fitness: null });
  });

  // completa população
  while (newPop.length < populationSize) {
    const p1 = tournamentSelection(population);
    const p2 = tournamentSelection(population);
    const childModel = crossover(p1.model, p2.model);
    mutate(childModel);
    newPop.push({ model: childModel, fitness: null });
  }

  return newPop;
}

// -------------------------------------------------
// SALVAR MELHOR MODELO GLOBAL
// -------------------------------------------------
async function saveGlobalBestModel(model, generation, fitness) {
  const modelName = `best-model-gen-${generation}-fitness-${fitness.toFixed(2)}`;
  const modelPath = path.join(modelsDir, modelName);

  try {
    await model.save(`file://${modelPath}`);
    console.log(`Melhor modelo salvo: geração ${generation}, fitness=${fitness.toFixed(2)}`);
  } catch (err) {
    console.error(`Erro ao salvar modelo:`, err);
  }
}

// -------------------------------------------------
// FUNÇÕES DE AVALIAÇÃO DOS INDIVÍDUOS
// (onde chamamos os Workers para rodar o "game" e
//  retornar o fitness de cada modelo)
// -------------------------------------------------
async function evaluatePopulation(generation) {
  return new Promise((resolve) => {
    let completed = 0;
    const total = population.length;

    population.forEach((individual, idx) => {
      // Serializa pesos
      const weightsArr = individual.model.getWeights().map(w => w.arraySync());

      // Cria um Worker
      const worker = new Worker(path.resolve(__dirname, './worker.js'), {
        workerData: {
          workerId: idx,
          modelWeights: weightsArr,
          currentGeneration: generation,
          maxGenerations
        }
      });

      // Ouvir mensagens do Worker
      worker.on('message', (msg) => {
        if (msg.type === 'fitness') {
          // Guardar fitness
          individual.fitness = msg.fitness;
          completed++;

          console.log(`Worker ${msg.workerId} => fitness: ${msg.fitness} / gen:${generation}`);

          if (completed === total) {
            resolve();
          }
        }

        if (msg.type === 'modelSaved') {
          console.log(
            `Worker ${msg.workerId} salvou um "melhor modelo" local: G=${msg.generation}, fit=${msg.fitness}`
          );
        }
      });

      worker.on('error', err => {
        console.error(`Erro no Worker ${idx}`, err);
        completed++;
        if (completed === total) {
          resolve();
        }
      });
    });
  });
}

// -------------------------------------------------
// LOOP PRINCIPAL: EXECUTAR GERAÇÕES
// -------------------------------------------------
async function runGenerations() {
  initializePopulation();

  for (let gen = 0; gen < maxGenerations; gen++) {
    console.log(`\n=== Geração ${gen + 1} ===`);

    // Avaliar
    await evaluatePopulation(gen);

    // Pega estatísticas
    const best = selectElite(population)[0];  // 1o da elite
    const bestInGen = best.fitness;
    const avg = population.reduce((acc, cur) => acc + cur.fitness, 0) / population.length;

    console.log(`> Best fitness: ${bestInGen.toFixed(2)} | Avg: ${avg.toFixed(2)}`);

    // Se for melhor que o global, salva
    if (bestInGen > overallBestFitness) {
      overallBestFitness = bestInGen;
      overallBestModel = await cloneModel(best.model);
      await saveGlobalBestModel(overallBestModel, gen + 1, overallBestFitness);
    }

    // Manda p/ front via socket.io (opcional)
    generationData.push({ generation: gen + 1, best: bestInGen, avg: avg });
    io.emit('generationData', generationData);

    // Nova pop
    const elite = selectElite(population);
    population = createNextGeneration(elite);
  }

  console.log('Treinamento concluído.');
  if (overallBestModel) {
    await saveGlobalBestModel(overallBestModel, 'final', overallBestFitness);
  }
  io.emit('done', { message: 'Treinamento concluído.' });
}

// -------------------------------------------------
// CLONE MODEL (ex.: para best global)
// -------------------------------------------------
async function cloneModel(originalModel) {
  const newModel = createModel();
  const originalWeights = originalModel.getWeights();
  const clonedWeights = originalWeights.map(w => w.clone());
  newModel.setWeights(clonedWeights);

  // compila se precisar
  newModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });
  return newModel;
}

// -------------------------------------------------
// INICIAR O GA
// -------------------------------------------------
runGenerations().catch(err => {
  console.error('Erro no runGenerations:', err);
});
