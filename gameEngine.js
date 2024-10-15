class GameEngine {
  constructor() {
    this.reset();
  }

  reset() {
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
    this.gravity = -0.6;
    this.jumpVelocity = 12;
    this.groundYPos = 0;
    this.time = Date.now();
    this.deltaTime = 0;
    this.distanceRan = 0;
    this.scoreCoefficient = 0.025;
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

    this.handleAction(action);
    this.updateTRex();
    this.updateObstacles();

    if (this.checkCollision()) {
      this.gameOver = true;
    } else {
      this.distanceRan += (this.currentSpeed * this.deltaTime) / this.msPerFrame;
      this.score = Math.floor(this.distanceRan * this.scoreCoefficient);

      if (this.currentSpeed < this.maxSpeed) {
        this.currentSpeed += this.acceleration * (this.deltaTime / this.msPerFrame);
      }
    }
  }

  handleAction(action) {
    if (action === 0 && !this.tRex.jumping && !this.tRex.ducking) {
      // Pular
      this.tRex.jumping = true;
      this.tRex.velocityY = this.jumpVelocity;
    } else if (action === 1 && !this.tRex.jumping) {
      // Abaixar
      this.tRex.ducking = true;
      this.tRex.height = 25;
      this.tRex.width = 59;
    } else if (action === 2) {
      // Cancelar abaixar
      if (this.tRex.ducking) {
        this.tRex.ducking = false;
        this.tRex.height = 47;
        this.tRex.width = 44;
      }
    }
  }

  updateTRex() {
    if (this.tRex.jumping) {
      this.tRex.velocityY += this.gravity;
      this.tRex.yPos += this.tRex.velocityY;

      if (this.tRex.yPos <= 0) {
        this.tRex.yPos = 0;
        this.tRex.jumping = false;
        this.tRex.velocityY = 0;
      }
    }
  }

  updateObstacles() {
    this.obstacles.forEach((obstacle) => {
      obstacle.xPos -= this.currentSpeed * (this.deltaTime / this.msPerFrame);
    });

    this.obstacles = this.obstacles.filter(
      (obstacle) => obstacle.xPos + obstacle.width > 0
    );

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
    const obstacleTypes = this.score >= 300 ? [
      // Variantes de cactos e pássaros após 300 pontos
      { width: 30, height: 35, yPos: 0 },
      { width: 40, height: 35, yPos: 0 },
      { width: 50, height: 35, yPos: 0 },
      { width: 70, height: 35, yPos: 0 },
      { width: 25, height: 45, yPos: 0 },
      { width: 45, height: 45, yPos: 0 },
      { width: 75, height: 45, yPos: 0 },
      { width: 30, height: 25, yPos: 30 }, // Pássaro voador baixo
      { width: 30, height: 25, yPos: 80 }, // Pássaro voador alto
    ] : [
      // Cactos antes dos 300 pontos
      { width: 30, height: 35, yPos: 0 },
      { width: 40, height: 35, yPos: 0 },
      { width: 50, height: 35, yPos: 0 },
      { width: 70, height: 35, yPos: 0 },
      { width: 25, height: 45, yPos: 0 },
      { width: 45, height: 45, yPos: 0 },
      { width: 75, height: 45, yPos: 0 },
    ];

    const obstacleType =
      obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];

    const obstacle = {
      xPos: this.dimensions.WIDTH,
      yPos: obstacleType.yPos,
      width: obstacleType.width,
      height: obstacleType.height,
      collisionBoxes: [{ x: 0, y: 0, width: obstacleType.width, height: obstacleType.height }],
    };

    this.obstacles.push(obstacle);
    this.nextObstacleDistance = null;
  }

  checkCollision() {
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
        y: this.dimensions.HEIGHT - tRex.height - tRex.yPos + tRexBox.y,
        width: tRexBox.width,
        height: tRexBox.height,
      };

      for (let obstacleBox of obstacle.collisionBoxes) {
        const adjObstacleBox = {
          x: obstacle.xPos + obstacleBox.x,
          y: this.dimensions.HEIGHT - obstacle.height - obstacle.yPos + obstacleBox.y,
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
    return {
      tRexX: this.tRex.xPos,
      tRexY: this.tRex.yPos,
      tRexVelocityY: this.tRex.velocityY,
      tRexOnGround: this.tRex.yPos === 0 ? 1 : 0,
      obstacleX: this.obstacles.length > 0 ? this.obstacles[0].xPos : null,
      obstacleY: this.obstacles.length > 0 ? this.obstacles[0].yPos : null,
      obstacleWidth: this.obstacles.length > 0 ? this.obstacles[0].width : null,
      obstacleHeight: this.obstacles.length > 0 ? this.obstacles[0].height : null,
      score: this.score,
      gameOver: this.gameOver,
      currentSpeed: this.currentSpeed,
    };
  }
}

module.exports = GameEngine;
