# Frozen Lake

## What is this application?

Frozen Lake is a simple application in which and elf (the agent) tries and obtain a present (the goal) while traversing through and environment. The reason for implementing multiple models here is to show how various Reinforcement Learning Algorithms work and what are there tendencies (NOTE: Some tendencies could be problem specific).

## Install Dependencies

Make sure you have the right downloads to run the models by running the code chunk in the **'InstallDependencies.ipynb'** file. Afterwards, you should be set!

## Where do I start?

A great place to start when it comes to looking at the various reinforment learning algorithms and the problem itself is by looking at the **'MainDisplay.ipynd'** file. In this file it will train up four common reinforcement learning models, run them all (meaning that you can see the elf trying to reach the present), and then see their policys (the path each algorithm takes) along with the model training progress if the model learns by taking actions in some sort of way.

## How do I check out an Individual Model?

To check out one model you can visit one of the 4 files, of which each holds a specific model.

- **'DynamicProgramming.ipynb'** This file holds the Dynamic Programming model.
- **'QLearning.ipynb'** This file holds the Q-learning model.
- **'SARSA.ipynb'** This file holds the SARSA model.
- **'EligibilityTraces.ipynb'** This file holds the Eligibility Traces model.

## Model Evalution

### Compare each Model

Seeing how each model compares

<!-- ![A plot showing the 3 action learning models](/images/frozen_lake_model_comparison.png){: width="500"} -->
<img src="/images/frozen_lake_model_comparison.png" alt="A plot showing the 3 action learning models" width="1000">

### Visualize the Current Environment

Seeing the given environment for each model

![The environment ploted](/images/frozen_lake_environment.png)

### Visualize each of the Model's Policies

See the Dynamic Programming Policy

![The Dynamic Programming Policy](/images/frozen_lake_Dynamic%20Programming_policy.png)

See the QLearning Policy

![The QLearning Policy](/images/frozen_lake_QLearning_policy.png)

See the SARSA Policy

![The SARSA Policy](/images/frozen_lake_SARSA_policy.png)

See the Eligibility Traces Policy

![The Eligibility Traces Policy](/images/frozen_lake_Eligibility%20Traces_policy.png)

## Description and Thoughts on each Model

### Dynamic Programming

Dynamic Programming is a method in reinforcement learning which tries and preplan so that the agent can make the best possible moves. It does this by iterating through every state and checking the reward of all given neighbors. Given enough sweeps (full scan of the environment), determined by the threshold, the policy will be updated with near perfect values for the agent to traverse the given environment.

Dynamic Programming is actually the fastest to run out of the rest listed here and i guess thats due to it not dealing with episods. I also guess that this is the case because the policy consists of only the states here and is not a QTable so it has 1/4 of the values to cover in comparision to the other methods.

### QLearning

QLearning is a method in reinforcement learning which tries and learn in a very greedy way. This means given a list of actions it will choose the one that has the most immediate reward and update the state action pair accordingly.

QLearning I would say is one of the best algorithms for this problem. Firstly, I think it's very intuitive for the problem and to solve it with a QTable. Secondly, I the greedy nature of this algorithm leads to the agent learning very quick on whats bad because if it falls into a hole, it will not do so again. Also the learning rate is constantly change which helps midigate the problem of making very greedy action to early on.

### SARSA

SARSA is a method in reinforcement learning which tries and learn in a somewhat greedy way. It's very similar to sarsa in the fact that they both have and agent learn based on a direct formula, but different in the way the agent learns. With SARSA the agent learns to not take as much risk, but still try and maximize the reward.

This is why some iterations of the SARSA model will walk around a group of holes even if it can walk through them no problem. Again, this is because the model learned to avoid an area of states instead of just avoiding a sertain state, leading to a more reserved algorithm in comparision to QLearning. This model also has a learning rate which is constantly changing which helps the algorithm be less greedy early on.

### Eligibility Traces

Eligibility Traces is a method in reinforcement learning which tries and learn by tracing the agents steps for a given episode (1 assigned to eact step) and through those 'traces' tries and update the given model for future iterations. There is a decay rate factor added to this trace every step so that we make sure that we are edging closer towards the reward.

This model out of the bunch here I would say has the hardest time mostly because it takes way longer to come up with a good solution. An example of this is QLearning might need to take 5000 iterations to learn present obtaining behavior, while it takes Eligibility Traces 25000 iterations to learn that same present obtaining behavior. This could be partially due to the low gamma rate, which prioritieses short term reward, but this is because larger number are not able to be computed for some reason. A little tweek may solve that problem, but this is the Eligibility Traces model for now.

## Order of Efficiency for Models in terms of the Frozen Lake

Dynamic Programming > QLearning > SARSA > Eligibility Traces
