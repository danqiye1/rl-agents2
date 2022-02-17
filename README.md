# Reinforcement Learning Agents

I hope that this repository will be my personal framework for Deep Reinforcement Learning in future. Of course this remains to be seen.

## Current WIP:
1. Deep Q Learning on Atari Pong.
2. Add `wandb` for experiment tracking.

## Running Experiments
Entry points to experiments are in `experiments` folder. 

To train a pong agent:
```bash
$ python experiments/pong.py --train
```

To continue training from a model:
```bash
$ python experiments/pong.py --train -l checkpoints/step-<step_num>
```

To evaluate a model:
```bash
$ python experiments/pong.py --eval -l checkpoints/step-<step_num>
```

## Things that could break DQN if not done properly:
1. Use PongDeterministic-v4, since this implemented frame skipping.
2. Make sure to preprocess the image correctly into 84 x 84 and stack 4 frames into 4 channels.
3. Learning rate must be as small as 0.00025. I missed a 0 once.
4. Make sure in `target = rewards + self.gamma * expected_v * (1 - done)`, the (1 - done) factor is there so as to account for rewards of terminal states. We should not add the value of next states (represented by `expected_v`) if we are already in terminal states.
5. Make sure in `loss = self.criterion(estimated_q, target.unsqueeze(1))` the target and estimated Q value has the same dimensions (might be caused by improper shape of rewards)
6. Use Kaiming Initialization, since CNN use ReLU activation.
7. Make sure policy model and target model are not referencing the same model object (deep copy).
8. Make sure we gather across the right dimension in `estimated_q = policy_q.gather(1, actions)`