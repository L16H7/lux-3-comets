# Multi-agent Reinforcement Learning solution for Lux AI Neurips 2024 Competition
https://www.kaggle.com/competitions/lux-ai-season-3/overview

## Solution
We choose multi-agent RL to tackle the problem. Our primary motivation is to study multiagent learning behavior and to explore the possibilities of MARL research.

## Early concepts
Our goal is to minimize implementing hardcoded rules to improve the agents. We are curious to see how well the agents can learn this challenging partially observable environments.</br></br>
The biggest chanllenge is that the relic fragments are always hidden from the observation. It can only be observed through points collected at a timestep. The straight-forward approach would be to track the points gained and solved the problem mathematically.

#### Experiment with RNN
Instead of using maths, we experiment meta reinforcement learning algorithm [1]. If the agents remember where they get the points (rewards for meta RL), the agents should learn the exact locations of relic fragments. Our experiment code is based on the xland-minigrid [2]. We give memory unit with RNN for each agent. The idea is to give each individual unit remembers and learns the optimal strategy.</br></br>
The observation are fed as 7x7 egocentric information channel and the actions are conditioned on observation embeddings and hidden states of RNN. The model parameters are shared for all the agents. During inference, the observation and memory are unique to each agent.</br></br>
We observe interesting behavior for some maps that are hard to find relics. Sometimes, only two or three agents find the relics in first match. For the following matches, only those agents go to the relic nodes where the other agents are searching for it.</br></br>
To address this, we add state observation. In LuxAI, the environment dimensions are fixed, 24x24. So, we can just take 24x24 observation instead of concatenating the individual observations which is common in multi-agent RL. That solves the problem of sharing the information across all agents.</br></br>
We modify RNN hidden state for state observation only. The idea is to offload the agents the ability to remember. State observation RNN should act as the queen of the hive who remembers what happens.</br></br>
In theory, we imagine it like Edge of Tomorrow [3] where the Omega remembers how the team gets defeated and come up with adaptive strategies to beat the opponent team in next matches. It would be pretty cool if it works.[4]</br></br>
Results are not good enough. The agents do not figure out the exact fragment positions. There is a lot of wiggling behavior where agents move a lot in fragment clusters because they do not know exaction reward positions. There is no signifant improvement after a certain training time. Our hypothesis is that agents cannot learn combat behavior when they do not have exact fragment positions. So, whoever moves to opponent territory loses because they do not have clear advantage predictions.

#### Integrating rule-based for first match
We take open-source rule-based agent by aDg4b [4]. We use it to play the first match. The idea is to get the fragment positions and feed them starting from match 2. The RL agents take over in match 2. We modify the experiment by feeding the ground-truth locations of fragments. That is where we notice significant improvements in mean points collected compared to RNN solutions.</br></br>
The points collected with RNN is max 200 per match where feeding the ground-truth fragments is recorded at 400 points per match.</br></br>
There is the point where we get to experiment with different architectures and hyperparameters [5]. 

## Breakthrough
At this point, I am on the verge of giving up. Failing experiments take a toll on my confidence. We decide to pivot to doing for the sake of learning. We determine to use graph neural network to model the communication structure between agents. We decide that we need to calculate fragment nodes manually this time. After that, we want to run one last time to make sure the code is correct. We no longer use rule-based agent for match 1. Since this is a test run, we remove state observation and experiment with agent-specific egocentric observations.  We make it 47x47 because the agent is at the center and we want the agent to see everything on the map. State encoder is no longer needed this way. The downside is about 75% of the observation is NULL at any timestep. It is meant to be a test so it does not matter if it performs well or not. It was in the first week of Feb. Then, it is miraculously performing well in the leaderboard. The position jumped from 150+ to 11-15 in a day of training.

## Final weeks
We try giving more information and adding more layers to the neural network. I do not pay attention to the reward structure. The reward structure is point difference. We find out that the agents did not learn to sacrifice the initial points gained. The other teams demonstrate the ability to ambush the fragment clusters from the fog. Our RL agents never learn that ability. We reason that it may be due to the reward structure. When we calculate reward for losing match, the reward is higher even though it is losing. e.g., the other team sacrifice for 20 timesteps, we are leading by  30-50 points and the other team wipe out all the agents and take the lead. By the reward structure, this is fine. We revise reward structure to weight more on winning or losing the match. Only less than 20 days left, we cannot explore much.

## Model Architecture
We use multiagent promixmal policy optimization [8]. The actor model parameters are shared for all the agents. Each agent needs own inference with own observation to get an action. The critic sees both teams. We decide this because the critic needs to know the complete situation of opponent teams without fog [9]. We share more detail in model_architecture.pdf [10]

## Learning points and future work
#### Performance metrics
We should have implemented all the metrics and monitoring before we start the training. It is not efficient to watch and analyse the replays. The match statistics are needed to see if the agents learn a particular strategy or not. We can track how the behavior evolves over time. I believe we should have every statistics we can think of.

#### Action masking
We have basic action masking. We expect RL agents will learn themselves without strict intervention. This is wrong in our case. We need to mask everything that does not make sense from the beginning. I believe adding rules that we are confident is benefitual. I forget to mask insufficient energy for sapping. They fail to learn that. So, the norm is not to have expectation that the agents will learn to not do silly actions. It has been studied action masking improves RL learning a lot [7]. At that point, it is the common wisdom.

#### Model architecture
I am solely responsible for experimenting with different architectures. My modeling skill is subpar. I believe the model architecture is the bottleneck. I try to make it work with vision transformer but in the end, I go with convolution neural network with residual connections. Near the end of the competition, I experiment with vision transformer with CNN teacher model. Vision transformer shows some promise but it is taking a lot longer. With limited compute budget, it is not feasible.

#### Agent coordination
Since all the agents decide the action themselves, there are weaknesses in attack and defense. One major weakness is we need to use action sampling. If we take max likely action, agents are not sapping a lot and the performance is signifcantly worse than sampling. Agents do not know which agents will be sapping and doing what. I try to solve it with communication action head but it does not work. In next season, I would like to explore communication protocol to coordinate actions better, possibly with graph neural networks.

## Conclusion
I like to thank everyone. I like to thank the creators of Lux AI and the supportive community. It has been fun. I wanted to win a prize. It was stressful and chanllenging. I had fun anyway. I learned a lot. I take this as a failure that helps me grow. I hope I will be mature enough just participate for fun and learning new things instead of pushing myself to win.</br></br>
I like to thank my lifelong mentor who teach me RL for many months. I learn RL a lot faster because of you. In the past months, I learn how to read research papers, how to ask interesting questions and how to trace the thoughts of researchers. I am glad I get this mentorship. I hope to share it with anyone who wants to pursue RL.</br></br>

## Notes
We share the github repository https://github.com/L16H7/lux-3-comets/. I am sorry for the messy code. I will clean the repository when I am available later.

## References and Footnotes
[1] Duan, Yan, et al. "Rl $^ 2$: Fast reinforcement learning via slow reinforcement learning." arXiv preprint arXiv:1611.02779 (2016) https://arxiv.org/abs/1611.02779
</br>
[2] Nikulin, Alexander, et al. "XLand-minigrid: Scalable meta-reinforcement learning environments in JAX." The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2024. https://github.com/dunnolab/xland-minigrid
</br>
[3] https://en.wikipedia.org/wiki/Edge_of_Tomorrow
</br>
[4] The reality is the ideas are hampered by my engineering, modelling and experimentation skills. I am not able to make the training works.
</br>
[5] https://www.kaggle.com/code/egrehbbt/relicbound-bot-for-lux-ai-s3-competition
</br>
[6] Unfortunately, I personally make a lot of mistakes and bugs at this point. The biggest mistake is the bug with sapping masking. The agents do not learn combat strategies because I mask a lot of possible sapping action space by mistake. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/556943
</br>
[7] Huang, Shengyi, and Santiago Ontañón. "A closer look at invalid action masking in policy gradient algorithms." arXiv preprint arXiv:2006.14171 (2020). https://arxiv.org/abs/2006.14171
</br>
[8] Yu, Chao, et al. "The surprising effectiveness of ppo in cooperative multi-agent games." Advances in neural information processing systems 35 (2022): 24611-24624. https://arxiv.org/abs/2103.01955
</br>
[9] https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
</br>
[10] https://github.com/L16H7/lux-3-comets/blob/master/model_architecture.pdf
