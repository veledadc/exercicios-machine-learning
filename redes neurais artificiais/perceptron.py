# Implementação Perceptron

import random

class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
		self.amostras = amostras
		self.saidas = saidas
		self.taxa_aprendizado = taxa_aprendizado
		self.epocas = epocas
		self.limiar = limiar
		self.n_amostras = len(amostras)
		self.n_atributos = len(amostras[0])
		self.pesos = []

	def treinar(self):

		for amostra in self.amostras:
			amostra.insert(0, -1)

		for i in range(self.n_atributos):
			self.pesos.append(random.random())

		self.pesos.insert(0, self.limiar)
		n_epocas = 0 # contador de épocas

		while True:

			erro = False # erro inicialmente inexiste

			for i in range(self.n_amostras):
				u = 0
				for j in range(self.n_atributos + 1):
					u += self.pesos[j] * self.amostras[i][j]
				y = self.sinal(u) # obtém a saída da rede

				# verifica se a saída da rede é diferente da saída desejada
				if y != self.saidas[i]:
					# calcula o erro
					erro_aux = self.saidas[i] - y
					# faz o ajuste dos pesos para cada elemento da amostra
					for j in range(self.n_atributos + 1):
						self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
					erro = True # o erro ainda existe

			n_epocas += 1 # incrementa o número de épocas

			# critério de parada
			if not erro or n_epocas > self.epocas:
				break

	def teste(self, amostra):
		amostra.insert(0, -1)
		u = 0
		for i in range(self.n_atributos + 1):
			u += self.pesos[i] * amostra[i]
		y = self.sinal(u)
		print('Classe: %d' % y)

	def degrau(self, u):
		if u >= 0:
			return 1
		return 0

	def sinal(self, u):
		if u >= 0:
			return 1
		return -1


# OR
'''
amostras = [[0, 0], [0, 1], [1, 0], [1, 1]]
saidas = [0, 1, 1, 1]
rede = Perceptron(amostras, saidas)
rede.treinar()
rede.teste([1, 1])
'''

# outro exemplo
'''
amostras = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2],
			[0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]
saidas = [1, -1, -1, 1]
rede = Perceptron(amostras, saidas)
rede.treinar()
rede.teste([0.5, 0.7, 0.1])
'''

# blue ou red?
amostras = [[0.72, 0.82], [0.91, -0.69],
				[0.46, 0.80],   [0.03, 0.93],
				[0.12, 0.25],   [0.96, 0.47],
				[0.8, -0.75],   [0.46, 0.98],
				[0.66, 0.24],   [0.72, -0.15],
				[0.35, 0.01],   [-0.16, 0.84],
				[-0.04, 0.68],  [-0.11, 0.1],
				[0.31, -0.96],   [0.0, -0.26],
				[-0.43, -0.65],  [0.57, -0.97],
				[-0.47, -0.03],  [-0.72, -0.64],
				[-0.57, 0.15],   [-0.25, -0.43],
				[0.47, -0.88],   [-0.12, -0.9],
				[-0.58, 0.62],   [-0.48, 0.05],
				[-0.79, -0.92],  [-0.42, -0.09],
				[-0.76, 0.65],   [-0.77, -0.76]]

saidas = [-1,-1,-1,-1,-1,-1, -1,-1,
			-1,-1,-1,-1,-1,1,1,1,1,
			1,1,1,1,1,1,1,1,1,1,1,1,1]
rede = Perceptron(amostras, saidas)
rede.treinar()
rede.teste([0.46, 0.80])