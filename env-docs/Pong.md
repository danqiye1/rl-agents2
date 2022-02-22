# Pong

## Environment Name
Use PongDeterministic-v4 to get the one with the skipped frames.

## Observation
Type: ndarray
dtype: uint8
dimensions: (210, 160, 3)

## Action Space
Type: Discrete(6) corresponding to ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

## Reward
Miss the paddle: Reward -1
Default: 0