# Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent Reinforcement Learning with General Utilities (2023)
Authors:

## To run:

1. Install required packages.
   ```bash
   pip install wandb
   pip install SuperSuit==3.6.0
   pip install pettingzoo==1.22.0
   pip install torch==1.13.1
   ```

2. Start training. See the beginning of '__main__' for a list of arguments it takes.
  
   ```bash
   python3 main_synthetic.py
   ```
3. To sync the results via wandb:
        
   ```bash
   python3 main_synthetic.py  --track
   ```

## Network Architecture
![Network architecture. Every agent trains a different set of weights and can only communicate
with their neighbors in a decentralized manner.](./readme_images/alg_flow.png)

## Experiment results
![Synthetic experiment on N agents passing a message from right to left. Reward depends on the agent's own action
and the next agent's state.](./readme_images/synthetic.png)

![How the magnitude of the RHS of the constraint affects each agent's likelihood of violating the
constraint.](./readme_images/constraint_rhs_value_constraint.png)
