/*************************************************
 * Exemplo: MultiGameEngine.js
 * 
 * - Suporta n Dinos simultâneos num só "mundo".
 * - Cada Dino tem:
 *    - xPos, yPos
 *    - velocityY
 *    - jumping, ducking
 *    - width, height
 *    - score
 *    - alive (bool)
 * - Obstáculos (this.obstacles) são compartilhados.
 *************************************************/
class MultiGameEngine {
    constructor(numDinos = 5) {
      this.numDinos = numDinos;
      this.dimensions = { WIDTH: 600, HEIGHT: 150 };
  
      this.gravity = -0.6;
      this.jumpVelocity = 12;
      this.msPerFrame = 1000 / 60; // 60 FPS
      this.acceleration = 0.001;
      this.currentSpeed = 6;
      this.maxSpeed = 13;
      this.scoreCoefficient = 0.025;
  
      this.reset();
    }
  
    /**
     * (Re)inicia completamente o "mundo":
     * - Dinos em posição inicial
     * - Obstáculos vazios
     * - Tempo zerado
     */
    reset() {
      this.dinos = [];
      for (let i = 0; i < this.numDinos; i++) {
        // Cada Dino é um objeto independente
        this.dinos.push({
          xPos: 50,
          yPos: 0,
          width: 44,
          height: 47,
          velocityY: 0,
          jumping: false,
          ducking: false,
          alive: true,          // se false, "morreu" (colisão)
          score: 0,
          distanceRan: 0,
        });
      }
  
      this.obstacles = [];
      this.gameTime = Date.now();
      this.deltaTime = 0;
      this.nextObstacleDistance = null;
      this.currentSpeed = 6; // reseta a velocidade
  
      // (Opcional) se quiser "gameOver" quando T-O-D-O-S morrerem
      this.allDinosDead = false;
    }
  
    /**
     * Atualiza TODOS os dinos e obstáculos.
     * Recebe um array `actions`, de tamanho = número de dinos,
     * para dizer a ação de cada dino.
     * Ex: actions[i] = 0 => pular, 1 => abaixar, 2 => nada
     */
    updateAllDinos(actions) {
      if (this.allDinosDead) return;
  
      const now = Date.now();
      this.deltaTime = now - (this.gameTime || now);
      this.gameTime = now;
  
      // 1) Aplica a ação em cada Dino
      for (let i = 0; i < this.numDinos; i++) {
        const dino = this.dinos[i];
        // Se dino já morreu, pula
        if (!dino.alive) continue;
  
        this.handleAction(dino, actions[i]);
      }
  
      // 2) Atualiza posição vertical de cada Dino (pulo / queda)
      for (let i = 0; i < this.numDinos; i++) {
        const dino = this.dinos[i];
        if (!dino.alive) continue;
  
        // Se está pulando
        if (dino.jumping) {
          dino.velocityY += this.gravity;
          dino.yPos += dino.velocityY;
  
          if (dino.yPos <= 0) {
            dino.yPos = 0;
            dino.jumping = false;
            dino.velocityY = 0;
          }
        }
      }
  
      // 3) Atualiza obstáculos
      this.updateObstacles();
  
      // 4) Check Collisions
      this.checkCollisionsAll();
  
      // 5) Se Dino está vivo => atualiza distância, score
      //    e acelera o speed do jogo
      let anyAlive = false;
      for (let i = 0; i < this.numDinos; i++) {
        const dino = this.dinos[i];
        if (dino.alive) {
          anyAlive = true;
          dino.distanceRan += (this.currentSpeed * this.deltaTime) / this.msPerFrame;
          dino.score = Math.floor(dino.distanceRan * this.scoreCoefficient);
        }
      }
  
      if (!anyAlive) {
        this.allDinosDead = true;
        return;
      }
  
      // 6) Acelera a speed
      if (this.currentSpeed < this.maxSpeed) {
        this.currentSpeed += this.acceleration * (this.deltaTime / this.msPerFrame);
      }
    }
  
    /**
     * Aplica a ação no Dino:
     * 0 => pular
     * 1 => abaixar (se não estiver pulando)
     * 2 => soltar/ficar em pé
     */
    handleAction(dino, action) {
      // Se Dino morto, ignora
      if (!dino.alive) return;
  
      if (action === 0 && !dino.jumping && !dino.ducking) {
        // pular
        dino.jumping = true;
        dino.velocityY = this.jumpVelocity;
      } else if (action === 1 && !dino.jumping && !dino.ducking) {
        // abaixar
        dino.ducking = true;
        dino.height = 25;
        dino.width = 59;
      } else if (action === 2) {
        // soltar se estiver abaixado
        if (dino.ducking) {
          dino.ducking = false;
          dino.height = 47;
          dino.width = 44;
        }
      }
    }
  
    /**
     * Gera/atualiza a posição de obstáculos no "mundo".
     * Idêntico ao seu GameEngine "updateObstacles", mas
     * mantém a lógica para todos os dinos ao mesmo tempo.
     */
    updateObstacles() {
      // Move cada obstáculo
      for (let obs of this.obstacles) {
        obs.xPos -= this.currentSpeed * (this.deltaTime / this.msPerFrame);
      }
  
      // remove obstáculos que saíram da tela
      this.obstacles = this.obstacles.filter(
        (obs) => obs.xPos + obs.width > 0
      );
  
      // define se precisa criar obstáculo
      if (this.shouldAddNewObstacle()) {
        this.addNewObstacle();
      }
    }
  
    shouldAddNewObstacle() {
      if (this.obstacles.length === 0) return true;
  
      const lastObstacle = this.obstacles[this.obstacles.length - 1];
  
      if (!this.nextObstacleDistance) {
        const minGap = (100 + this.currentSpeed * 10) * 1.2;
        const maxGap = (200 + this.currentSpeed * 14) * 1.2;
        this.nextObstacleDistance =
          Math.floor(Math.random() * (maxGap - minGap + 1)) + minGap;
      }
  
      return (
        lastObstacle.xPos + lastObstacle.width <
        this.dimensions.WIDTH - this.nextObstacleDistance
      );
    }
  
    addNewObstacle() {
      // Exemplo de variedade
      const obstacleTypes = [
        { width: 30, height: 35, yPos: 0 },
        { width: 40, height: 35, yPos: 0 },
        { width: 50, height: 35, yPos: 0 },
        { width: 30, height: 25, yPos: 30 },
        { width: 30, height: 25, yPos: 60 },
      ];
  
      const obstacleType =
        obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
  
      const newObs = {
        xPos: this.dimensions.WIDTH,
        yPos: obstacleType.yPos,
        width: obstacleType.width,
        height: obstacleType.height,
        collisionBoxes: [
          { x: 0, y: 0, width: obstacleType.width, height: obstacleType.height },
        ],
      };
  
      this.obstacles.push(newObs);
      this.nextObstacleDistance = null;
    }
  
    /**
     * Verifica colisão de TODOS os dinos com os obstáculos.
     * Quem colidir => alive = false
     */
    checkCollisionsAll() {
      for (let dino of this.dinos) {
        if (!dino.alive) continue;
  
        for (let obs of this.obstacles) {
          if (this.isColliding(dino, obs)) {
            // Dino morre
            dino.alive = false;
            break;
          }
        }
      }
    }
  
    /**
     * Lógica de colisão 2D: 
     * Ajuste se quiser as "collisionBoxes"
     */
    isColliding(dino, obstacle) {
      // Ajuste as boxes
      // Exemplo: Dino tem 2 boxes:
      const dinoCollisionBoxes = [
        { x: 1, y: 1, width: 30, height: 35 },
        { x: 5, y: 35, width: 20, height: 10 },
      ];
  
      // Se dino estiver abaixado, poderia trocar as boxes, se quiser
      // simplificar, use só 1 bounding box ou algo similar.
  
      // Convertendo cada box pra coords absolutos
      for (let dinoBox of dinoCollisionBoxes) {
        const adjDinoBox = {
          x: dino.xPos + dinoBox.x,
          y: this.dimensions.HEIGHT - dino.height - dino.yPos + dinoBox.y,
          width: dinoBox.width,
          height: dinoBox.height,
        };
  
        for (let obsBox of obstacle.collisionBoxes) {
          const adjObsBox = {
            x: obstacle.xPos + obsBox.x,
            y: this.dimensions.HEIGHT - obstacle.height - obstacle.yPos + obsBox.y,
            width: obsBox.width,
            height: obsBox.height,
          };
  
          if (this.boxCompare(adjDinoBox, adjObsBox)) {
            return true;
          }
        }
      }
      return false;
    }
  
    boxCompare(a, b) {
      return !(
        a.x > b.x + b.width ||
        a.x + a.width < b.x ||
        a.y > b.y + b.height ||
        a.y + a.height < b.y
      );
    }
  
    /**
     * Retorna "todas" as infos de display:
     * - cada dino (pos, se vivo, score)
     * - os obstáculos
     */
    getDisplayState() {
      return {
        dinos: this.dinos.map((d, idx) => ({
          xPos: d.xPos,
          yPos: d.yPos,
          width: d.width,
          height: d.height,
          alive: d.alive,
          score: d.score,
        })),
        obstacles: this.obstacles.map(o => ({
          xPos: o.xPos,
          yPos: o.yPos,
          width: o.width,
          height: o.height,
        })),
        currentSpeed: this.currentSpeed,
        allDinosDead: this.allDinosDead,
      };
    }
  
    /**
     * Se você quiser extrair "estado" para cada dino, para
     * passar ao modelo. Ex: retorna array de states normalizados
     * para cada dino.
     */
    getStatesForAllDinos() {
      const states = [];
      for (let i = 0; i < this.numDinos; i++) {
        const d = this.dinos[i];
        // se quiser normalizar, faça aqui
        const firstObs = this.obstacles[0] || null;
        states.push({
          tRexY: d.yPos,
          velocityY: d.velocityY,
          obstacleX: firstObs ? firstObs.xPos : null,
          obstacleW: firstObs ? firstObs.width : 0,
          obstacleH: firstObs ? firstObs.height : 0,
          alive: d.alive,
        });
      }
      return states;
    }
  
    /**
     * Exemplo: saber se ainda existe dino vivo
     */
    hasAnyAlive() {
      return this.dinos.some(d => d.alive);
    }
  }
  
  // Exporta a classe (caso esteja em Node.js ou bundler):
  // module.exports = MultiGameEngine;
  
  /*************************************************
   * Exemplo de uso (front-end ou Node):
   *************************************************/
  
  // const game = new MultiGameEngine(5);
  // game.updateAllDinos([0,1,2,0,2]); // 5 dinos => 5 ações
  // const displayState = game.getDisplayState();
  // console.log(displayState);
  
  