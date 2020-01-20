# Redes Neurais utilizando PyBrain

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# passa as dimensões dos vetores de entrada e do objetivo
dataset = SupervisedDataSet(2, 1)

dataset.addSample([1,1], [0])
dataset.addSample([1,0], [1])
dataset.addSample([0,1], [1])
dataset.addSample([0,0], [0])

network = buildNetwork(dataset.indim, 4, dataset.outdim, bias=True)
trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

'''
for epoch in range(1000): # treina por 1000 épocas
	trainer.train()
'''

trainer.trainEpochs(1000)
'''
	treinar até a convergência: trainer.trainUntilConvergence
'''

test_data = SupervisedDataSet(2, 1)
test_data.addSample([1,1], [0])
test_data.addSample([1,0], [1])
test_data.addSample([0,1], [1])
test_data.addSample([0,0], [0])
trainer.testOnData(test_data, verbose=True)