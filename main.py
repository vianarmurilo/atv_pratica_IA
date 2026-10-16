  # pyright: basic

import copy
import csv   
import os
import random
import shutil

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

gym.register_envs(ale_py)


ENV_ID = "ALE/Freeway-v5"
TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 3
TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2

MODO_TESTE = False
if MODO_TESTE:
    TAMANHO_POPULACAO = 6
    NUM_GERACOES = 3

EXPERIMENTOS = {
    "A": (0.01, 0.05),
    "B": (0.10, 0.20),
    "C": (0.40, 0.50),
}


def criar_ambiente(render_mode=None):
    return gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)


class Agente:
    def __init__(self):
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0

    def agir(self, estado):
        estado = estado.astype(np.float32) / 255.0
        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2
        return int(np.argmax(saida))



def avaliar(agente, render_mode=None):
    env = criar_ambiente(render_mode)
    estado, info = env.reset()             
    recompensa_total = 0.0
    terminou = False

    while not terminou:
        acao = agente.agir(estado)          
        estado, reward, terminated, truncated, info = env.step(acao)
        recompensa_total += reward          
        terminou = terminated or truncated  

    env.close()
    return recompensa_total                 


def avaliar_aleatorio():
    
    env = criar_ambiente()
    env.reset()
    total = 0.0
    terminou = False
    while not terminou:
        acao = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(acao)
        total += reward
        terminou = terminated or truncated
    env.close()
    return total



def criar_populacao():
    return [Agente() for _ in range(TAMANHO_POPULACAO)]


def selecionar(populacao):
    ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)
    metade = len(ordenada) // 2
    return ordenada[:metade]


def mutar(agente, taxa=TAXA_MUTACAO, intensidade=INTENSIDADE_MUTACAO):
    for nome in ["W1", "b1", "W2", "b2"]:
        pesos = getattr(agente, nome)
        sorteados = np.random.rand(*pesos.shape) < taxa
        ruido = np.random.randn(*pesos.shape) * intensidade
        setattr(agente, nome, pesos + sorteados * ruido)


def nova_geracao(populacao, taxa=TAXA_MUTACAO, intensidade=INTENSIDADE_MUTACAO):
    ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)

    
    nova = [copy.deepcopy(a) for a in ordenada[:NUM_ELITE]]

    
    pais = selecionar(populacao)
    while len(nova) < TAMANHO_POPULACAO:
        pai = random.choice(pais)
        filho = copy.deepcopy(pai)
        filho.fitness = 0.0
        mutar(filho, taxa, intensidade)
        nova.append(filho)

    return nova



def salvar_csv(historico, caminho):
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["geracao", "melhor", "media", "pior"])
        escritor.writeheader()
        escritor.writerows(historico)


def salvar_grafico(historico, caminho, titulo):
    geracoes = [h["geracao"] for h in historico]
    plt.figure(figsize=(8, 5))
    plt.plot(geracoes, [h["melhor"] for h in historico], label="Melhor fitness")
    plt.plot(geracoes, [h["media"] for h in historico], label="Fitness médio")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.savefig(caminho, dpi=150)
    plt.close()


def rodar_evolucao(taxa, intensidade, semente=42):
    np.random.seed(semente)
    random.seed(semente)

    populacao = criar_populacao()
    historico = []
    melhor_agente = None

    for geracao in range(NUM_GERACOES):
        for agente in populacao:
            agente.fitness = avaliar(agente)

        valores = [a.fitness for a in populacao]
        melhor = max(valores)
        media = float(np.mean(valores))
        pior = min(valores)
        historico.append({"geracao": geracao, "melhor": melhor,
                          "media": media, "pior": pior})
        print(f"Geração {geracao:03d} | Melhor: {melhor:.2f} | "
              f"Média: {media:.2f} | Pior: {pior:.2f}")

        
        atual = max(populacao, key=lambda a: a.fitness)
        if melhor_agente is None or atual.fitness > melhor_agente.fitness:
            melhor_agente = copy.deepcopy(atual)

        if geracao < NUM_GERACOES - 1:
            populacao = nova_geracao(populacao, taxa, intensidade)

    return historico, melhor_agente


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)

    
    baseline = [avaliar_aleatorio() for _ in range(10)]
    print(f"\nBASELINE aleatório (10 episódios): média = {np.mean(baseline):.2f}\n")

   
    resultados = {}
    for nome, (taxa, intensidade) in EXPERIMENTOS.items():
        print(f"\n=== Experimento {nome}: taxa={taxa} intensidade={intensidade} ===")
        historico, melhor = rodar_evolucao(taxa, intensidade)
        salvar_csv(historico, f"resultados/fitness_{nome}.csv")
        salvar_grafico(historico, f"resultados/fitness_{nome}.png",
                       f"Experimento {nome} (taxa={taxa}, intensidade={intensidade})")
        resultados[nome] = (historico, melhor)

    shutil.copy("resultados/fitness_B.csv", "resultados/fitness.csv")
    shutil.copy("resultados/fitness_B.png", "resultados/fitness.png")

    
    melhor_b = resultados["B"][1]
    notas = [avaliar(melhor_b) for _ in range(10)]
    print(f"\nMelhor agente (B), 10 episódios: média = {np.mean(notas):.2f}")
    print(f"Agente aleatório,    10 episódios: média = {np.mean(baseline):.2f}")

   
    avaliar(melhor_b, render_mode="human")