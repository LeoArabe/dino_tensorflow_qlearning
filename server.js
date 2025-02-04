// server.js
const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const { exec } = require('child_process');
const app = express();
const http = require('http');
const server = http.createServer(app);
const socketIo = require('socket.io');
const io = socketIo(server);
const port = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'ai-public')));

// Endpoint para salvar o melhor modelo (GA)
const BEST_MODEL_FILE = path.join(__dirname, 'best-models', 'best_model.json');
// Endpoint para salvar o best score de todos os tempos
const BEST_SCORE_FILE = path.join(__dirname, 'best-models', 'best_score_of_all_time.json');

// Garantir que a pasta 'best-models' exista
if (!fs.existsSync(path.join(__dirname, 'best-models'))) {
  fs.mkdirSync(path.join(__dirname, 'best-models'), { recursive: true });
  console.log('[server] Pasta "best-models" criada.');
}

app.post('/api/save-best-model', (req, res) => {
  const { bestModel, bestFitness } = req.body;
  if (!bestModel || bestFitness === undefined) {
    return res.status(400).json({ message: 'Dados insuficientes para salvar o modelo.' });
  }
  const modelData = {
    bestFitness,
    bestModel,
    time: Date.now()
  };
  fs.writeFile(BEST_MODEL_FILE, JSON.stringify(modelData, null, 2), err => {
    if (err) {
      console.error('Erro ao salvar o modelo:', err);
      return res.status(500).json({ message: 'Erro ao salvar o modelo.' });
    }
    console.log(`[server] Melhor modelo salvo com fitness: ${bestFitness}`);
    res.status(200).json({ message: 'Modelo salvo com sucesso.' });
  });
});

app.get('/api/best-model', (req, res) => {
  if (!fs.existsSync(BEST_MODEL_FILE)) {
    return res.status(404).json({ message: 'Nenhum modelo salvo ainda.' });
  }
  fs.readFile(BEST_MODEL_FILE, 'utf8', (err, data) => {
    if (err) {
      console.error('Erro ao ler o modelo:', err);
      return res.status(500).json({ message: 'Erro ao ler o modelo.' });
    }
    try {
      const modelData = JSON.parse(data);
      res.status(200).json(modelData);
    } catch (parseErr) {
      console.error('Erro ao parsear o modelo:', parseErr);
      res.status(500).json({ message: 'Dados do modelo inválidos.' });
    }
  });
});

// Endpoints para o best score de todos os tempos
app.post('/api/save-best-score', (req, res) => {
  const { bestScore } = req.body;
  if (bestScore === undefined) {
    return res.status(400).json({ message: 'Best score não fornecido.' });
  }
  const scoreData = {
    bestScore,
    time: Date.now()
  };
  fs.writeFile(BEST_SCORE_FILE, JSON.stringify(scoreData, null, 2), err => {
    if (err) {
      console.error('Erro ao salvar o best score:', err);
      return res.status(500).json({ message: 'Erro ao salvar o best score.' });
    }
    console.log(`[server] Best score salvo: ${bestScore}`);
    res.status(200).json({ message: 'Best score salvo com sucesso.' });
  });
});

app.get('/api/best-score', (req, res) => {
  if (!fs.existsSync(BEST_SCORE_FILE)) {
    return res.status(404).json({ message: 'Nenhum best score salvo ainda.' });
  }
  fs.readFile(BEST_SCORE_FILE, 'utf8', (err, data) => {
    if (err) {
      console.error('Erro ao ler o best score:', err);
      return res.status(500).json({ message: 'Erro ao ler o best score.' });
    }
    try {
      const scoreData = JSON.parse(data);
      res.status(200).json(scoreData);
    } catch (parseErr) {
      console.error('Erro ao parsear o best score:', parseErr);
      res.status(500).json({ message: 'Dados do best score inválidos.' });
    }
  });
});

server.listen(port, () => {
  console.log(`Servidor rodando na porta ${port}`);
});
