# Frozen Lake

### What is this application?

Frozen Lake is a simple application in which and elf (the agent) tries and obtain a present (the goal) while traversing through and environment. The reason for implementing multiple models here is to show how various Reinforcement Learning Algorithms work and what are there tendencies (NOTE: Some tendencies could be problem specific).

### Install Dependencies

Make sure you have the right downloads to run the models by running the code chunk in the **'InstallDependencies.ipynb'** file. Afterwards, you should be set!

### Where do I start?

A great place to start when it comes to looking at the various reinforment learning algorithms and the problem itself is by looking at the **'MainDisplay.ipynd'** file. In this file it will train up four common reinforcement learning models, run them all (meaning that you can see the elf trying to reach the present), and then see their policys (the path each algorithm takes) along with the model training progress if the model learns by taking actions in some sort of way.

### How do I check out an Individual Model?

To check out one model you can visit one of the 4 files, of which each holds a specific model.
**'DynamicProgramming.ipynb'** This file holds the Dynamic Programming model.
**'QLearning.ipynb'** This file holds the Q-learning model.
**'SARSA.ipynb'** This file holds the SARSA model.
**'EligibilityTraces.ipynb'** This file holds the Eligibility Traces model.
