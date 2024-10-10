class GameEngine {
  constructor() {
    this.reset();
  }

  reset() {
    // Inicialize o estado do jogo
    this.tRex = {
      xPos: 50,
      yPos: 0,
      jumping: false,
      ducking: false,
      velocityY: 0,
      width: 44,
      height: 47,
      collisionBoxes: [
        { x: 1, y: 1, width: 30, height: 35 },
        { x: 5, y: 35, width: 20, height: 10 },
      ],
    };

    this.obstacles = [];
    this.score = 0;
    this.gameOver = false;
    this.currentSpeed = 6;
    this.maxSpeed = 13;
    this.acceleration = 0.001;
    this.gravity = 0.6;
    this.groundYPos = 0;
    this.time = Date.now();
    this.deltaTime = 0;
    this.distanceRan = 0;
    this.scoreCoefficient = 0.1; // Coeficiente ajustado
    this.msPerFrame = 1000 / 60;
    this.nextObstacleDistance = null;
    this.dimensions = {
      WIDTH: 600,
      HEIGHT: 150,
    };
  }

  update(action) {
    if (this.gameOver) return;

    const now = Date.now();
    this.deltaTime = now - (this.time || now);
    this.time = now;

    // Tratar a ação do jogador
    this.handleAction(action);

    // Atualizar o T-Rex
    this.updateTRex();

    // Atualizar os obstáculos
    this.updateObstacles();

    // Verificar colisões
    if (this.checkCollision()) {
      this.gameOver = true;
    } else {
      // Atualizar a distância percorrida
      this.distanceRan += (this.currentSpeed * this.deltaTime) / this.msPerFrame;

      // Atualizar o score
      this.score = Math.floor(this.distanceRan * this.scoreCoefficient);

      // Aumentar a velocidade do jogo gradualmente
      if (this.currentSpeed < this.maxSpeed) {
        this.currentSpeed += this.acceleration * (this.deltaTime / this.msPerFrame);
      }
    }
  }

  handleAction(action) {
    // 0: pular, 1: abaixar, 2: nada
    if (action === 0 && !this.tRex.jumping) {
      this.tRex.jumping = true;
      this.tRex.velocityY = -12; // Velocidade inicial do pulo ajustada
    } else if (action === 1 && !this.tRex.jumping) {
      this.tRex.ducking = true;
      this.tRex.height = 25; // Altura reduzida ao abaixar
    } else {
      this.tRex.ducking = false;
      this.tRex.height = 47; // Altura normal
    }
  }

  updateTRex() {
    if (this.tRex.jumping) {
      this.tRex.velocityY += this.gravity;
      this.tRex.yPos += this.tRex.velocityY;

      if (this.tRex.yPos >= 0) {
        this.tRex.yPos = 0;
        this.tRex.jumping = false;
        this.tRex.velocityY = 0;
      }
    }
  }

  updateObstacles() {
    // Atualizar obstáculos existentes
    this.obstacles.forEach((obstacle) => {
      obstacle.xPos -= this.currentSpeed * (this.deltaTime / this.msPerFrame);
    });

    // Remover obstáculos que saíram da tela
    this.obstacles = this.obstacles.filter(
      (obstacle) => obstacle.xPos + obstacle.width > 0
    );

    // Adicionar novos obstáculos conforme necessário
    if (this.shouldAddNewObstacle()) {
      this.addNewObstacle();
    }
  }

  shouldAddNewObstacle() {
    // Adicionar obstáculos somente após o score atingir 40
    if (this.score < 40) {
      return false;
    }

    if (this.obstacles.length === 0) return true;

    const lastObstacle = this.obstacles[this.obstacles.length - 1];

    if (!this.nextObstacleDistance) {
      const minGap = 100; // Gap mínimo ajustado
      const maxGap = 300; // Gap máximo ajustado
      this.nextObstacleDistance =
        Math.floor(Math.random() * (maxGap - minGap + 1)) + minGap;
    }

    return (
      lastObstacle.xPos + lastObstacle.width <
      this.dimensions.WIDTH - this.nextObstacleDistance
    );
  }

  addNewObstacle() {
    // Defina tipos de obstáculos com diferentes tamanhos e caixas de colisão
    const obstacleTypes = [
      {
        width: 17,
        height: 35,
        yPos: 0,
        collisionBoxes: [
          { x: 0, y: 7, width: 5, height: 27 },
          { x: 4, y: 0, width: 6, height: 34 },
          { x: 10, y: 4, width: 7, height: 14 },
        ],
      },
      // Adicione outros tipos de obstáculos conforme necessário
    ];

    const obstacleType =
      obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];

    const obstacle = {
      xPos: this.dimensions.WIDTH,
      yPos: obstacleType.yPos,
      width: obstacleType.width,
      height: obstacleType.height,
      collisionBoxes: obstacleType.collisionBoxes,
    };

    this.obstacles.push(obstacle);
    this.nextObstacleDistance = null;
  }

  checkCollision() {
    // Verificar colisões entre o T-Rex e os obstáculos
    for (let obstacle of this.obstacles) {
      if (this.isColliding(this.tRex, obstacle)) {
        return true;
      }
    }
    return false;
  }

  isColliding(tRex, obstacle) {
    for (let tRexBox of tRex.collisionBoxes) {
      const adjTrexBox = {
        x: tRex.xPos + tRexBox.x,
        y: tRex.yPos + tRexBox.y,
        width: tRexBox.width,
        height: tRexBox.height,
      };

      for (let obstacleBox of obstacle.collisionBoxes) {
        const adjObstacleBox = {
          x: obstacle.xPos + obstacleBox.x,
          y: obstacle.yPos + obstacleBox.y,
          width: obstacleBox.width,
          height: obstacleBox.height,
        };

        if (this.boxCompare(adjTrexBox, adjObstacleBox)) {
          return true;
        }
      }
    }
    return false;
  }

  boxCompare(boxA, boxB) {
    return !(
      boxA.x > boxB.x + boxB.width ||
      boxA.x + boxA.width < boxB.x ||
      boxA.y > boxB.y + boxB.height ||
      boxA.y + boxA.height < boxB.y
    );
  }

  getState() {
    // Retorne o estado atual do jogo
    return {
      tRexX: this.tRex.xPos,
      tRexY: this.tRex.yPos,
      tRexVelocityY: this.tRex.velocityY,
      obstacles: this.obstacles.map((obstacle) => ({
        xPos: obstacle.xPos,
        yPos: obstacle.yPos,
        width: obstacle.width,
        height: obstacle.height,
      })),
      score: this.score,
      gameOver: this.gameOver,
    };
  }
}

module.exports = GameEngine;
