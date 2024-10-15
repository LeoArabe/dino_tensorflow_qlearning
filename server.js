const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { Worker } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const os = require('os');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Variáveis globais para armazenar dados para exibição
let generationData = [];

// Servir arquivos estáticos da pasta 'public'
app.use(express.static(path.join(__dirname, 'public')));

const numWorkers = os.cpus().length - 1; // Usar o número de CPUs disponíveis
const populationSize = numWorkers;
const maxGenerations = 300;

// Definição de eliteSize
const eliteSize = Math.max(1, Math.floor(populationSize * 0.2)); // Pelo menos 1

let population = [];
let generation = 0;

// Função para criar um novo modelo com pesos aleatórios
function createRandomModel() {
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

// Função para determinar o número de instâncias por worker com base na geração atual
function getNumInstancesPerWorker(currentGeneration, maxGenerations) {
  const initialInstances = 1; // Ajuste conforme necessário
  const finalInstances = 1; // Ajuste conforme necessário
  
  const ratio = currentGeneration / maxGenerations; // Progresso da geração
  
  // Interpolar entre o valor inicial e final
  return Math.floor(initialInstances - (initialInstances - finalInstances) * ratio);
}

// Função para inicializar a população
function initializePopulation() {
  population = [];
  for (let i = 0; i < populationSize; i++) {
    const model = createRandomModel();
    population.push({
      model,
      fitness: null,
    });
  }
}

// Função para serializar os pesos do modelo
function serializeWeights(model) {
  return model.getWeights().map(t => t.arraySync());
}

// Função para desserializar os pesos
function deserializeWeights(serializedWeights) {
  return serializedWeights.map(w => tf.tensor(w));
}

// Função para obter a taxa de mutação adaptativa
function getMutationRate(currentGeneration, maxGenerations) {
  const initialRate = 0.3; // Taxa de mutação inicial alta
  const finalRate = 0.01;  // Taxa de mutação final baixa
  return initialRate - ((initialRate - finalRate) * (currentGeneration / maxGenerations));
}

// Função para obter a taxa de crossover adaptativa
function getCrossoverRate(currentGeneration, maxGenerations) {
  const initialRate = 0.9; // Taxa de crossover inicial alta
  const finalRate = 0.6;   // Taxa de crossover final mais baixa
  return initialRate - ((initialRate - finalRate) * (currentGeneration / maxGenerations));
}

// Função para obter o tamanho do torneio adaptativo
function getTournamentSize(currentGeneration, maxGenerations) {
  const initialSize = 2; // Tamanho inicial do torneio
  const finalSize = 5;   // Tamanho final do torneio
  return Math.floor(initialSize + ((finalSize - initialSize) * (currentGeneration / maxGenerations)));
}

// Função para seleção dos melhores indivíduos (elitismo)
function selectElite(population) {
  return population
    .sort((a, b) => b.fitness - a.fitness)
    .slice(0, eliteSize);
}

// Função para crossover entre dois pais
function crossover(parent1, parent2, crossoverRate) {
  const childModel = createRandomModel();
  const parent1Weights = parent1.model.getWeights();
  const parent2Weights = parent2.model.getWeights();

  const childWeights = [];
  for (let i = 0; i < parent1Weights.length; i++) {
    const weight1 = parent1Weights[i];
    const weight2 = parent2Weights[i];

    // Gerar máscara de crossover com base na taxa de crossover adaptativa
    const shape = weight1.shape;
    const randMatrix = tf.randomUniform(shape, 0, 1);
    const mask = tf.lessEqual(randMatrix, crossoverRate);

    const weight1Masked = tf.mul(weight1, mask);
    const weight2Masked = tf.mul(weight2, tf.logicalNot(mask));

    const childWeight = tf.add(weight1Masked, weight2Masked);
    childWeights.push(childWeight);
  }

  childModel.setWeights(childWeights);
  return childModel;
}

// Função para mutação com taxa adaptativa
function mutate(model, mutationRate) {
  const weights = model.getWeights();
  const mutatedWeights = weights.map(weight => {
    const shape = weight.shape;
    const dtype = weight.dtype;

    // Gerar máscara de mutação com base na taxa adaptativa
    const mutationMask = tf.randomUniform(shape, 0, 1).lessEqual(mutationRate);
    const mutationValues = tf.randomNormal(shape, 0, 0.1);

    const mutatedWeight = weight.add(mutationValues.mul(mutationMask.cast(dtype)));
    return mutatedWeight;
  });

  model.setWeights(mutatedWeights);
  return model;
}

// Função para seleção por torneio com pressão variável
function tournamentSelection(population, tournamentSize) {
  const tournament = [];
  for (let i = 0; i < tournamentSize; i++) {
    const randomIndex = Math.floor(Math.random() * population.length);
    tournament.push(population[randomIndex]);
  }
  tournament.sort((a, b) => b.fitness - a.fitness);
  return tournament[0]; // Retorna o melhor do torneio
}

// Função para criar nova geração
function createNextGeneration(elite, currentGeneration) {
  const newPopulation = [];

  // Preservar a elite (elitismo)
  for (let i = 0; i < elite.length; i++) {
    newPopulation.push({
      model: elite[i].model,
      fitness: null,
    });
  }

  const mutationRate = getMutationRate(currentGeneration, maxGenerations);
  const crossoverRate = getCrossoverRate(currentGeneration, maxGenerations);
  const tournamentSize = getTournamentSize(currentGeneration, maxGenerations);

  // Gerar novos indivíduos através de crossover e mutação
  while (newPopulation.length < populationSize) {
    const parent1 = tournamentSelection(population, tournamentSize);
    const parent2 = tournamentSelection(population, tournamentSize);

    let childModel = crossover(parent1, parent2, crossoverRate);
    childModel = mutate(childModel, mutationRate);

    newPopulation.push({
      model: childModel,
      fitness: null,
    });
  }

  return newPopulation;
}

// Função para avaliar a população
function evaluatePopulation(currentGeneration) {
  return new Promise((resolve) => {
    let workersFinished = 0;
    const totalIndividuals = population.length;
    const numInstancesPerWorker = getNumInstancesPerWorker(currentGeneration, maxGenerations);

    for (let i = 0; i < population.length; i++) {
      const individual = population[i];

      // Serializar os pesos do modelo para enviar ao worker
      const modelWeights = serializeWeights(individual.model);

      const worker = new Worker(path.resolve(__dirname, './worker.js'), {
        workerData: {
          workerId: i,
          numInstances: numInstancesPerWorker,
          modelWeights: modelWeights,
          currentGeneration: currentGeneration,
          maxGenerations: maxGenerations,
        },
      });

      worker.on('message', (message) => {
        if (message.type === 'fitness') {
          individual.fitness = message.fitness;
          workersFinished++;

          console.log(`Worker ${message.workerId} retornou fitness: ${message.fitness}`);

          // Calcular o progresso em porcentagem
          const progressPercentage = ((workersFinished / totalIndividuals) * 100).toFixed(2);
          console.log(`Progresso da geração ${currentGeneration + 1}: ${progressPercentage}%`);

          // Enviar o progresso para o front-end via Socket.IO
          io.emit('generationProgress', {
            generation: currentGeneration + 1,
            progress: progressPercentage,
          });

          if (workersFinished === population.length) {
            resolve();
          }
        } else if (message.type === 'gameState') {
          // Enviar o estado do jogo para o cliente via Socket.IO
          io.emit('gameState', {
            workerId: message.workerId,
            gameState: message.gameState,
          });
        }
      });

      worker.on('error', (err) => {
        console.error(`Erro no worker ${i}:`, err);
        workersFinished++;

        if (workersFinished === population.length) {
          resolve();
        }
      });

      worker.on('exit', (code) => {
        if (code !== 0) {
          console.error(`Worker ${i} saiu com código ${code}`);
        }
      });
    }
  });
}

// Função para salvar o melhor modelo em um diretório
async function saveBestModel(model, generation, fitness) {
  const modelDir = `./models/generation_${generation}`;
  if (!fs.existsSync('./models')) {
    fs.mkdirSync('./models');
  }
  // Salva o modelo completo no diretório especificado
  await model.save(`file://${modelDir}`);

  console.log(`Melhor modelo salvo em ${modelDir} com fitness ${fitness}`);
}

// Função principal para executar as gerações
async function runGenerations() {
  initializePopulation();

  let overallBestIndividual = null;

  for (generation = 0; generation < maxGenerations; generation++) {
    console.log(`\nIniciando geração ${generation + 1}`);

    // Avaliar a população
    await evaluatePopulation(generation);

    // Seleção
    const elite = selectElite(population);
    console.log(`Melhor fitness da geração ${generation + 1}: ${elite[0].fitness}`);

    // Atualizar o melhor indivíduo geral
    if (!overallBestIndividual || elite[0].fitness > overallBestIndividual.fitness) {
      overallBestIndividual = {
        model: elite[0].model,
        fitness: elite[0].fitness,
      };

      // Salvar o melhor modelo encontrado até agora
      await saveBestModel(overallBestIndividual.model, generation + 1, overallBestIndividual.fitness);
    }

    // Coletar dados para exibição
    const avgFitness = population.reduce((sum, ind) => sum + ind.fitness, 0) / population.length;
    generationData.push({
      generation: generation + 1,
      bestFitness: elite[0].fitness,
      avgFitness: avgFitness,
    });

    // Emitir dados para o cliente via Socket.IO
    io.emit('generationData', generationData);

    // Criar nova geração
    population = createNextGeneration(elite, generation);
  }

  console.log('Algoritmo genético concluído.');

  // Salvar o melhor modelo final
  await saveBestModel(overallBestIndividual.model, 'final', overallBestIndividual.fitness);
}

// Iniciar o servidor na porta 3000
const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Servidor rodando na porta ${PORT}`);
});

// Iniciar o algoritmo genético após o servidor estar rodando
runGenerations();

// Rota para servir o arquivo HTML principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
