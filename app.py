# =====================================================
# APP WEB DE IA (STREAMLIT) - DO ZERO
# =====================================================

import math
import json
import streamlit as st

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuronio:
    def __init__(self):
        self.pesos = []
        self.bias = 0

    def forward(self, entradas):
        soma = self.bias
        for x, w in zip(entradas, self.pesos):
            soma += x * w
        return sigmoid(soma)

    def from_dict(self, data):
        self.pesos = data["pesos"]
        self.bias = data["bias"]

class Camada:
    def __init__(self, n_neuronios):
        self.neuronios = [Neuronio() for _ in range(n_neuronios)]

    def forward(self, entradas):
        return [n.forward(entradas) for n in self.neuronios]

    def from_dict(self, data):
        for n, d in zip(self.neuronios, data):
            n.from_dict(d)

class RedeNeural:
    def __init__(self, arquitetura):
        self.camadas = []
        for i in range(1, len(arquitetura)):
            self.camadas.append(Camada(arquitetura[i]))

    def forward(self, entradas):
        for camada in self.camadas:
            entradas = camada.forward(entradas)
        return entradas

    def carregar(self, arquivo):
        with open(arquivo) as f:
            modelo = json.load(f)
        for camada, dados in zip(self.camadas, modelo["camadas"]):
            camada.from_dict(dados)

st.set_page_config(page_title="Minha IA", layout="centered")
st.title("🤖 Minha IA (do zero)")

rede = RedeNeural([2, 3, 1])
rede.carregar("modelo_ia.json")

x1 = st.number_input("Entrada 1", 0.0, 1.0, 0.0, 0.1)
x2 = st.number_input("Entrada 2", 0.0, 1.0, 0.0, 0.1)

if st.button("Executar IA"):
    saida = rede.forward([x1, x2])[0]
    decisao = 1 if saida >= 0.5 else 0

    st.success(f"Saída da IA: {saida:.4f}")
    st.info(f"Decisão final: {decisao}")
