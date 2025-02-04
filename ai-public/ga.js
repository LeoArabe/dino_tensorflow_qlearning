// ga.js – Lógica do GA e classes Dino e MultiDinoEngine

window.testMode = window.testMode || false;

// Classe Dino
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
    this.action = 2; // 0 = pular, 1 = agachar, 2 = nada
    this.turnsSurvived = 0; // contador de turnos sobrevividos
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

// Cria um "cérebro" aleatório (simples com 15 pesos: 5 para cada ação)
function createRandomBrain() {
  const weights = [];
  for (let i = 0; i < 15; i++) {
    weights.push(Math.random() * 2 - 1);
  }
  return { weights };
}

// Classe MultiDinoEngine: gerencia vários dinos e obstáculos
class MultiDinoEngine {
  constructor(numDinos) {
    this.dimensions = { width: 800, height: 200 };
    this.gravity = -0.6;
    this.currentSpeed = 6;
    this.maxSpeed = 13;
    this.acceleration = 0.001;
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
    this.distanceRan += this.currentSpeed;
    let aliveCount = 0;
    for (let dino of this.dinos) {
      if (!dino.alive) continue;
      dino.turnsSurvived++;
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
      return true; // todos morreram
    }
    return false;
  }

  handleAction(dino, action) {
    if (action === 0 && !dino.jumping && !dino.ducking) {
      dino.jumping = true;
      dino.vy = 12;
    } else if (action === 1) {
      if (!dino.jumping) {
        dino.ducking = true;
        dino.height = 25;
        dino.width = 59;
      } else {
        // Se o dino está pulando e decide agachar, encurta o pulo
        if (dino.vy > 0) {
          dino.vy = 0;
        } else {
          dino.vy -= 2;
        }
      }
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
    // Não gera obstáculos até que uma distância mínima seja percorrida
    if (this.distanceRan < 300) return false;
    if (this.obstacles.length === 0) return true;
    if (!this.nextObstacleDist) {
      const minGap = 250;
      const maxGap = 450;
      this.nextObstacleDist = Math.floor(Math.random() * (maxGap - minGap + 1)) + minGap;
    }
    const last = this.obstacles[this.obstacles.length - 1];
    return (last.xPos + last.width < this.dimensions.width - this.nextObstacleDist);
  }

  addNewObstacle() {
    // Define os tipos de obstáculos básicos
    const obstacleTypes = [
      { type: 'ground', width: 30, height: 35, yPos: 0 },
      { type: 'aerial', width: 40, height: 30, yPos: 30 },
      { type: 'passive', width: 35, height: 35, yPos: 0 }
    ];
    // Se a distância percorrida for >= 10000, adiciona o obstáculo "high"
    if (this.distanceRan >= 10000) {
      // Para simular um pássaro voador:
      // - altura: 1/3 do valor anterior (40 em vez de 120)
      // - yPos: definido para que o obstáculo seja desenhado suspenso (por exemplo, 60)
      obstacleTypes.push({ type: 'high', width: 40, height: 40, yPos: 60 });
    }
    
    const rand = Math.random();
    let chosen;
    if (obstacleTypes.length === 3) {
      if (rand < 0.4) {
        chosen = obstacleTypes[0]; // ground
      } else if (rand < 0.8) {
        chosen = obstacleTypes[1]; // aerial
      } else {
        chosen = obstacleTypes[2]; // passive
      }
    } else {
      // Quando o obstáculo "high" está disponível (4 tipos)
      if (rand < 0.3) {
        chosen = obstacleTypes[0]; // ground
      } else if (rand < 0.55) {
        chosen = obstacleTypes[1]; // aerial
      } else if (rand < 0.8) {
        chosen = obstacleTypes[2]; // passive
      } else {
        chosen = obstacleTypes[3]; // high
      }
    }
    this.obstacles.push({
      xPos: this.dimensions.width,
      yPos: chosen.yPos,
      width: chosen.width,
      height: chosen.height,
      type: chosen.type
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
      const overlap = !(dinoBox.x2 < obsBox.x1 || dinoBox.x1 > obsBox.x2 ||
                        dinoBox.y2 < obsBox.y1 || dinoBox.y1 > obsBox.y2);
      if (overlap) {
        if (obs.type === 'ground' && dino.action === 0) continue;
        else if (obs.type === 'aerial' && dino.action === 1) continue;
        else if (obs.type === 'passive' && dino.action === 2) continue;
        // Para obstáculo "high": se o dino NÃO estiver pulando, ele consegue passar
        else if (obs.type === 'high' && !dino.jumping) continue;
        else return true;
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
    this.currentSpeed = 6;
    this.allDead = false;
    if (newDinos) {
      this.dinos = newDinos;
    } else {
      this.dinos = this.dinos.map(() => new Dino());
    }
  }
}

// Funções do GA
function selectElite(dinos, elitePercent) {
  const eliteCount = Math.max(1, Math.floor(dinos.length * (elitePercent / 100)));
  return [...dinos].sort((a, b) => b.fitness - a.fitness).slice(0, eliteCount);
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

function createNextGeneration(dinos, popSize, elitePercent) {
  const elite = selectElite(dinos, elitePercent);
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

// Expor no escopo global para uso em main.js
window.MultiDinoEngine = MultiDinoEngine;
window.selectElite = selectElite;
window.createNextGeneration = createNextGeneration;
