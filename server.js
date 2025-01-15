/************************************************** 
 * server.js (Node + Express)
 **************************************************/
const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const openBrowser = require('open'); // Para abrir o navegador automaticamente

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json()); // Para interpretar JSON no corpo das requisições

// Servir a pasta "public" como estática
app.use(express.static(path.join(__dirname, 'ai-public')));

// Rota principal -> index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'ai-public', 'index.html'));
});

// Caminho para o melhor modelo global
const BEST_MODEL_FILE = path.join(__dirname, 'best-models', 'best_model_of_all_time.json');

// Garantir que a pasta 'best-models' exista
const modelsDir = path.join(__dirname, 'best-models');
if (!fs.existsSync(modelsDir)) {
  fs.mkdirSync(modelsDir, { recursive: true });
  console.log('[server] Pasta "best-models" criada.');
}

// Rota para salvar o melhor modelo global
app.post('/save-model', (req, res) => {
  const { modelData, fitness, bestAllTime } = req.body;

  // Validar os dados recebidos
  if (!modelData || fitness == null) {
    return res.status(400).json({ message: 'Dados insuficientes para salvar o modelo.' });
  }

  let model;
  try {
    model = JSON.parse(modelData);
  } catch (err) {
    return res.status(400).json({ message: 'Dados do modelo inválidos.' });
  }

  // Montar o objeto a ser salvo
  const modelToSave = {
    fitness: parseFloat(fitness),
    brain: model
  };

  let filename;
  if (bestAllTime) {
    // Se for o melhor de todos os tempos, sobrescreve o arquivo fixo
    filename = 'best_model_of_all_time.json';
  } else {
    // Caso contrário, usa um nome com timestamp (opcional, mas não será usado atualmente)
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    filename = `best_model_${timestamp}.json`;
  }

  const filepath = path.join(modelsDir, filename);

  // Salvar o modelo como JSON de forma assíncrona
  fs.writeFile(filepath, JSON.stringify(modelToSave, null, 2), (err) => {
    if (err) {
      console.error('Erro ao salvar o modelo:', err);
      return res.status(500).json({ message: 'Erro ao salvar o modelo.' });
    }

    console.log(`[server] Modelo salvo: ${filename} | Fitness: ${fitness.toFixed(2)}`);
    res.status(200).json({ message: 'Modelo salvo com sucesso.', filename });
  });
});

// Rota para obter o melhor modelo global
app.get('/best-model', (req, res) => {
  if (!fs.existsSync(BEST_MODEL_FILE)) {
    return res.status(404).json({ message: 'Nenhum modelo salvo ainda.' });
  }

  fs.readFile(BEST_MODEL_FILE, 'utf8', (err, data) => {
    if (err) {
      console.error('Erro ao ler o melhor modelo:', err);
      return res.status(500).json({ message: 'Erro ao ler o modelo.' });
    }

    try {
      const model = JSON.parse(data);
      res.status(200).json(model);
    } catch (parseErr) {
      console.error('Erro ao parsear o modelo:', parseErr);
      res.status(500).json({ message: 'Dados do modelo inválidos.' });
    }
  });
});

// Inicia servidor
app.listen(PORT, () => {
  console.log(`[server] Rodando em http://localhost:${PORT}`);
  
  // Abrir no navegador automaticamente (opcional)
  try {
    openBrowser(`http://localhost:${PORT}`);
  } catch (err) {
    console.log('Falha ao abrir o navegador:', err.message);
  }
});
