// testEngine.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// 1) Pasta raiz com os modelos
const modelsRootDir = path.resolve(__dirname, 'models');

/**
 * Função para listar todos os subdiretórios de "models" que possuam um "model.json" dentro.
 * Caso o nome do arquivo seja diferente, ajuste a variável `modelJsonFilename`.
 */
function listAllModelFolders() {
  // Se você gerou o modelo com outro nome, ex.: "model-best-worker-0/model.json",
  //  então basta mudar para 'model.json' (ou 'mode.json', etc.)
  const modelJsonFilename = 'model.json'; 

  // Lê o conteúdo da pasta "models", pegando apenas os subdiretórios.
  const allDirs = fs.readdirSync(modelsRootDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  // Filtra apenas as pastas que contêm o arquivo "model.json"
  const validFolders = allDirs.filter(dirName => {
    const fullPath = path.join(modelsRootDir, dirName, modelJsonFilename);
    return fs.existsSync(fullPath);
  });

  return validFolders;
}

// 2) Obtém todos os diretórios com "model.json" (ou outro nome que você usar)
const allModelFolders = listAllModelFolders();

console.log(`Encontrados ${allModelFolders.length} modelos na pasta "models".`);
console.log(allModelFolders);

// 3) Servir "engineRealTime.html" + pastas de modelos
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'engineRealTime.html'));
});

// Servir a própria pasta atual (para pegar engineRealTime.html, JS, CSS, etc.)
app.use('/', express.static(__dirname));

// Servir a pasta "models" recursivamente
app.use('/models', express.static(modelsRootDir));

// Subir o servidor na porta desejada
const PORT = 3100; 
server.listen(PORT, () => {
  console.log(`Servidor testEngine rodando na porta ${PORT}`);
  openBrowsersForAllModels();
});

/**
 * Função que abre N abas (uma para cada pasta de modelo encontrada) em modo NÃO headless.
 * Ajuste se quiser abrir em headless, etc.
 */
async function openBrowsersForAllModels() {
  if (allModelFolders.length === 0) {
    console.log('Nenhum modelo encontrado. Não há abas para abrir.');
    return;
  }

  // Launch do Puppeteer
  const browser = await puppeteer.launch({
    headless: false,
    args: [
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding'
    ]
  });

  // Para cada pasta que possui "model.json", abrimos uma aba no Chrome.
  for (const folderName of allModelFolders) {
    // Monta a URL para passar ?modelFolder=models/<folderName>
    // Assim, no front-end (engineRealTime.html) você pode acessar o param
    // e carregar: tf.loadLayersModel(modelFolder + '/model.json');
    const pageUrl = `http://localhost:${PORT}/TestEngineInterface.html?modelFolder=models/${folderName}`;
    console.log(`Abrindo aba para modelo [${folderName}]: ${pageUrl}`);

    const page = await browser.newPage();
    await page.goto(pageUrl);
  }

  // Se você quiser fechar o browser após abrir, chame:
  // await browser.close();
}
