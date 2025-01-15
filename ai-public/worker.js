/**************************************************
 * worker.js
 **************************************************/
self.onmessage = function(e) {
    const { dinos, obstacles, generation } = e.data;
    let bestFitness = -Infinity;
    let bestModel = null;
  
    const updatedDinos = dinos.map(dino => {
      if (!dino.alive) return dino;
  
      // Decisão da ação baseada na rede neural (placeholder: aleatório)
      dino.action = decideAction(dino, obstacles);
  
      // Incrementar fitness
      dino.fitness += 1;
  
      // Verificar se este é o melhor dino localmente
      if (dino.fitness > bestFitness) {
        bestFitness = dino.fitness;
        bestModel = dino.brain;
      }
  
      return dino;
    });
  
    // Enviar os dados atualizados de volta para o front-end
    self.postMessage({
      updatedDinos,
      generation,
      bestFitness,
      bestModel
    });
  };
  
  // Função de decisão de ação (Placeholder: aleatório)
  function decideAction(dino, obstacles) {
    // Aqui você deve implementar a lógica baseada na rede neural
    // Para simplificação, estamos usando uma escolha aleatória
    return Math.floor(Math.random() * 3); // 0 = pular, 1 = abaixar, 2 = nada
  }
  