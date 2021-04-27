# "Towards Deep Learning Models Resistant to Adversarial Attacks" Re-Evaluation

Code used to re-evaluate the Projected Gradient Descent (PGD) statistics given in "Towards Deep Learning Models Resistant to Adversarial Attacks".

Credit to the original paper.

# Results

MNIST Attack PGD k=40 randomrestart=1
natural: 99.17%
adversarial: 0.00%
avg nat loss: 0.0315
avg adv loss: 46.6018

MNIST Defense PGD k=40 randomrestart=1
natural: 98.53%
adversarial: 94.23%
avg nat loss: 0.0409
avg adv loss: 0.1763

MNIST Defense PGD k=100 randomrestart=1
natural: 98.53%
adversarial: 92.70%
avg nat loss: 0.0409
avg adv loss: 0.2206

MNIST Defense CW k=40 randomrestart=1
natural: 98.53%
adversarial: 94.31%
avg nat loss: 0.0409
avg adv loss: 0.1704

All of the above have epsilon=0.3 and step size=0.01

CIFAR10 Attack PGD steps=7
natural: 95.01%
adversarial: 0.00%
avg nat loss: 0.2084
avg adv loss: 42.0510

CIFAR10 Defense PGD steps=7
natural: 87.14%
adversarial: 49.64%
avg nat loss: 0.4592
avg adv loss: 2.9006

CIFAR10 Defense PGD steps=20
natural: 87.14%
adversarial: 45.66%
avg nat loss: 0.4592
avg adv loss: 3.2901

CIFAR10 Defense CW steps=30
natural: 87.14%
adversarial: 46.54%
avg nat loss: 0.4592
avg adv loss: 3.2096

All of the above have epsilon=8 and step size=2.0