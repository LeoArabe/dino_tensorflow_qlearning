# Dino Q-Learning

[Portugues](#portugues) | [English](#english)

---

## Portugues

### Sobre
Clone do jogo Chrome Dino com um agente de IA que aprende a jogar usando Q-Learning e TensorFlow.js. O modelo treina observando o estado do jogo e aprende automaticamente quando pular ou abaixar.

### Tecnologias
- JavaScript
- TensorFlow.js (GPU)
- Puppeteer (automacao do navegador)
- Express + Socket.IO
- Node.js

### Como rodar
```bash
npm install
npm start
```

O treinamento sera iniciado automaticamente. O modelo treinado e salvo na pasta `model/`.

### Como funciona
1. O Puppeteer abre o jogo no navegador
2. O agente observa o estado do jogo (distancia e tamanho dos obstaculos)
3. O modelo de Q-Learning decide a acao (pular, abaixar ou nada)
4. O agente aprende com recompensas e penalidades a cada partida

### Status
Projeto funcional com modelo treinado.

---

## English

### About
Chrome Dino game clone with an AI agent that learns to play using Q-Learning and TensorFlow.js. The model trains by observing the game state and automatically learns when to jump or duck.

### Tech Stack
- JavaScript
- TensorFlow.js (GPU)
- Puppeteer (browser automation)
- Express + Socket.IO
- Node.js

### How to run
```bash
npm install
npm start
```

Training will start automatically. The trained model is saved in the `model/` folder.

### How it works
1. Puppeteer opens the game in the browser
2. The agent observes the game state (distance and size of obstacles)
3. The Q-Learning model decides the action (jump, duck or nothing)
4. The agent learns from rewards and penalties each round

### Status
Functional project with trained model.
