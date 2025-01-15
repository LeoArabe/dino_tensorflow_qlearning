/************************************************** 
 * ga.js — Lógica do GA + Classes Dino e Engine
 **************************************************/

/** Classe Dino */
class Dino {
    constructor(brain = null) {
      this.x = 50;
      this.y = 0;
      this.vy = 0;
      this.width = 44;
      this.height = 47;
      this.jumping = false;
      this.ducking = false;
      this.alive = true;
      this.fitness = 0;
      this.brain = brain || createRandomBrain();
      this.action = 2; // 0 = pular, 1 = duck, 2 = nada
    }
  
    update(gravity) {
      if (!this.alive) return;
      if (this.jumping) {
        this.vy += gravity;
        this.y += this.vy;
        if (this.y <= 0) {
          this.y = 0;
          this.jumping = false;
          this.vy = 0;
        }
      }
    }
  }
  
  /** Cria um "cérebro" aleatório (simples). */
  function createRandomBrain() {
    const weights = [];
    for (let i = 0; i < 20; i++) {
      weights.push(Math.random() * 2 - 1);
    }
    return { weights };
  }
  
  /** Classe MultiDinoEngine: gerencia vários dinos simultâneos. */
  class MultiDinoEngine {
    constructor(numDinos) {
      this.dimensions = { width: 800, height: 200 };
      this.gravity = -0.6;
      this.currentSpeed = 6; // Ajustado para dificultar conforme solicitado
      this.maxSpeed = 13;
      this.acceleration = 0.001; // Ajustado para dificultar conforme solicitado
      this.obstacles = [];
      this.nextObstacleDist = null;
      this.distanceRan = 0;
      this.allDead = false;
  
      this.dinos = [];
      for (let i = 0; i < numDinos; i++) {
        this.dinos.push(new Dino());
      }
    }
  
    updateAll() {
      this.updateObstacles();
      let aliveCount = 0;
      for (let dino of this.dinos) {
        if (!dino.alive) continue;
        aliveCount++;
        this.handleAction(dino, dino.action);
        dino.update(this.gravity);
        if (this.checkCollision(dino)) {
          dino.alive = false;
        } else {
          dino.fitness += this.currentSpeed; 
        }
      }
  
      if (this.currentSpeed < this.maxSpeed) {
        this.currentSpeed += this.acceleration;
      }
  
      if (aliveCount === 0 && !this.allDead) {
        this.allDead = true;
        return true; // todos mortos
      }
      return false; // ainda tem dinos vivos
    }
  
    handleAction(dino, action) {
      if (action === 0 && !dino.jumping && !dino.ducking) {
        dino.jumping = true;
        dino.vy = 12;
      } else if (action === 1 && !dino.jumping) {
        dino.ducking = true;
        dino.height = 25;
        dino.width = 59;
      } else if (action === 2 && dino.ducking) {
        dino.ducking = false;
        dino.height = 47;
        dino.width = 44;
      }
    }
  
    updateObstacles() {
      this.obstacles.forEach(obs => {
        obs.xPos -= this.currentSpeed;
      });
      this.obstacles = this.obstacles.filter(o => o.xPos + o.width > 0);
      if (this.shouldAddNewObstacle()) {
        this.addNewObstacle();
      }
    }
  
    shouldAddNewObstacle() {
      if (this.obstacles.length === 0) return true;
      const last = this.obstacles[this.obstacles.length - 1];
      if (!this.nextObstacleDist) {
        const minGap = 250; // Aumentado para dificultar
        const maxGap = 450; // Aumentado para dificultar
        this.nextObstacleDist = Math.floor(Math.random() * (maxGap - minGap + 1)) + minGap;
      }
      return (last.xPos + last.width < this.dimensions.width - this.nextObstacleDist);
    }
  
    addNewObstacle() {
      const obsTypes = [
        { width: 30, height: 35, yPos: 0 },
        { width: 35, height: 35, yPos: 0 },
        { width: 40, height: 35, yPos: 0 },
        { width: 50, height: 35, yPos: 0 },
      ];
      const oType = obsTypes[Math.floor(Math.random() * obsTypes.length)];
      this.obstacles.push({
        xPos: this.dimensions.width,
        yPos: oType.yPos,
        width: oType.width,
        height: oType.height
      });
      this.nextObstacleDist = null;
    }
  
    checkCollision(dino) {
      for (let obs of this.obstacles) {
        const dinoBox = {
          x1: dino.x,
          y1: this.dimensions.height - dino.height - dino.y,
          x2: dino.x + dino.width,
          y2: this.dimensions.height - dino.y
        };
        const obsBox = {
          x1: obs.xPos,
          y1: this.dimensions.height - obs.height - obs.yPos,
          x2: obs.xPos + obs.width,
          y2: this.dimensions.height - obs.yPos
        };
        if (!(dinoBox.x2 < obsBox.x1 || dinoBox.x1 > obsBox.x2 ||
              dinoBox.y2 < obsBox.y1 || dinoBox.y1 > obsBox.y2)) {
          return true;
        }
      }
      return false;
    }
  
    getStateForDino(dino) {
      const firstObs = this.obstacles[0] || null;
      const distToObs = firstObs ? (firstObs.xPos - dino.x) : 600;
      const normY = (dino.y + 100) / 100;
      const normVY = dino.vy / 20;
      const normDist = distToObs / 600;
      const normW = firstObs ? firstObs.width / 50 : 0;
      const normH = firstObs ? firstObs.height / 50 : 0;
      return [normY, normVY, normDist, normW, normH];
    }
  
    reset(newDinos) {
      this.obstacles = [];
      this.nextObstacleDist = null;
      this.distanceRan = 0;
      this.currentSpeed = 5; // Resetar para valor inicial ajustado
      this.allDead = false;
      if (newDinos) {
        this.dinos = newDinos;
      } else {
        this.dinos = this.dinos.map(() => new Dino());
      }
    }
  }
  
  /*****************************************************************
   * GA Functions (seleção, crossover, mutação) - SIMPLIFICADOS
   *****************************************************************/
  function selectElite(dinos, eliteSize) {
    return [...dinos].sort((a, b) => b.fitness - a.fitness).slice(0, eliteSize);
  }
  
  function tournamentSelection(dinos, size = 3) {
    let best = null;
    for (let i = 0; i < size; i++) {
      const r = Math.floor(Math.random() * dinos.length);
      const candidate = dinos[r];
      if (!best || candidate.fitness > best.fitness) {
        best = candidate;
      }
    }
    return best;
  }
  
  function crossover(brainA, brainB) {
    const child = { weights: [] };
    for (let i = 0; i < brainA.weights.length; i++) {
      child.weights.push(Math.random() < 0.5 ? brainA.weights[i] : brainB.weights[i]);
    }
    return child;
  }
  
  function mutate(brain, rate = 0.02) {
    for (let i = 0; i < brain.weights.length; i++) {
      if (Math.random() < rate) {
        brain.weights[i] += (Math.random() * 0.4 - 0.2);
      }
    }
  }
  
  function createNextGeneration(dinos, popSize) {
    const eliteSize = Math.floor(popSize * 0.05);
    const elite = selectElite(dinos, eliteSize);
    const newPop = elite.map(e => {
      const newBrain = { weights: [...e.brain.weights] };
      return new Dino(newBrain);
    });
    while (newPop.length < popSize) {
      const pA = tournamentSelection(dinos);
      const pB = tournamentSelection(dinos);
      const childBrain = crossover(pA.brain, pB.brain);
      mutate(childBrain, 0.02);
      newPop.push(new Dino(childBrain));
    }
    return newPop;
  }
  
  // Expor no escopo global (para main.js)
  window.MultiDinoEngine = MultiDinoEngine;
  window.selectElite = selectElite;
  window.createNextGeneration = createNextGeneration;
  