Absolutely. A good way to approach a **custom 2048 environment for reinforcement learning (RL) from scratch** is to treat the project as two separate pieces:

1. **The game/environment** — deterministic 2048 rules, state representation, rewards, termination.
2. **The RL system** — agent, neural network/value function, experience collection, training, evaluation.

The important part is designing the environment so that it is correct, reproducible, and easy for an RL algorithm to interact with.

## 1. What you are actually building

At the highest level, your system looks like this:

```text
                 ┌────────────────────┐
                 │    RL Agent        │
                 │                    │
                 │ Neural Network     │
                 │ + Policy / Value   │
                 └─────────┬──────────┘
                           │
                     action ∈ {0,1,2,3}
                           │
                           ▼
                 ┌────────────────────┐
                 │  Custom 2048 Env   │
                 │                    │
                 │ 4×4 board          │
                 │ Move logic         │
                 │ Tile spawning      │
                 │ Reward             │
                 │ Terminal condition │
                 └─────────┬──────────┘
                           │
                    next_state, reward
                           │
                           ▼
                 ┌────────────────────┐
                 │ Experience /       │
                 │ Training Pipeline  │
                 └────────────────────┘
```

The agent repeatedly performs:

$$
s_t \rightarrow a_t \rightarrow (r_t,s_{t+1})
$$

where:

* \(s_t\) = current 2048 board
* \(a_t\) = move chosen by the agent
* \(r_t\) = reward from the move
* \(s_{t+1}\) = resulting board

The goal is to learn a policy

$$
\pi(a|s)
$$

that chooses good actions for each board state.

---

# 2. Start by defining 2048 mathematically

Before writing any RL code, define the game itself.

A standard board is:

$$
B \in \mathbb{N}^{4\times4}
$$

For example:

```text
0   0   2   0
0   4   0   0
2   2   4   0
0   0   0   0
```

Here `0` means an empty cell.

The available actions are:

```text
0 = UP
1 = RIGHT
2 = DOWN
3 = LEFT
```

You could use strings instead:

```python
actions = ["up", "right", "down", "left"]
```

but integer actions are generally more convenient for RL.

---

# 3. Define the environment API

A very useful design is to make your environment behave similarly to common RL environments.

Conceptually:

```python
env = Game2048()

state = env.reset()

while not done:
    action = agent.choose_action(state)

    next_state, reward, done, info = env.step(action)

    agent.learn(state, action, reward, next_state, done)

    state = next_state
```

Your environment therefore needs at least:

```python
reset()
step(action)
```

I'd also implement:

```python
render()
valid_actions()
is_game_over()
clone()
```

So your environment might look like:

```python
class Game2048:

    def __init__(self):
        self.board = None
        self.score = 0

    def reset(self):
        ...

    def step(self, action):
        ...

    def render(self):
        ...

    def valid_actions(self):
        ...

    def is_game_over(self):
        ...
```

This separation is extremely important.

Your RL algorithm shouldn't need to know **how 2048 works internally**.

---

# 4. Board initialization

At the beginning of an episode, create an empty board:

```text
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
```

Then randomly place two tiles.

Standard 2048 generally generates:

* `2` with probability 0.9
* `4` with probability 0.1

So:

```python
def spawn_tile(self):
    empty_cells = get_empty_cells()

    if not empty_cells:
        return False

    position = random.choice(empty_cells)

    value = 2 if random.random() < 0.9 else 4

    self.board[position] = value

    return True
```

After initialization:

```text
0 0 0 2
0 0 0 0
0 4 0 0
0 0 0 0
```

This is your initial state \(s_0\).

---

# 5. The most important part: movement

The movement system is where many custom 2048 implementations accidentally introduce bugs.

Consider:

```text
2 2 4 4
```

Moving left should produce:

```text
4 8 0 0
```

not:

```text
8 4 0 0
```

The reason is that merges occur **from the direction of movement** and each tile can participate in only one merge during a move.

A clean methodology is to reduce every move to a **one-dimensional row operation**.

---

# 6. Implement one-dimensional compression

Suppose we have:

```text
[2, 0, 2, 4]
```

First remove zeros:

```text
[2, 2, 4]
```

This is the compression operation.

```python
def compress(row):
    return [x for x in row if x != 0]
```

Then perform merging.

```text
[2, 2, 4]
 ↓
[4, 4]
```

Then pad with zeros:

```text
[4, 4, 0, 0]
```

So the overall operation is:

```text
input
  ↓
compress
  ↓
merge
  ↓
compress/pad
  ↓
output
```

---

# 7. The merge algorithm

A robust implementation is:

```python
def merge_left(row):

    tiles = [x for x in row if x != 0]

    result = []
    reward = 0

    i = 0

    while i < len(tiles):

        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:

            merged = tiles[i] * 2

            result.append(merged)

            reward += merged

            i += 2

        else:

            result.append(tiles[i])

            i += 1

    result += [0] * (4 - len(result))

    return result, reward
```

For:

```text
[2, 2, 4, 4]
```

you get:

```text
[4, 8, 0, 0]
```

and reward:

$$
4+8=12
$$

---

# 8. Why the `i += 2` matters

This is a subtle but crucial part of 2048.

Consider:

```text
2 2 2 2
```

The correct result is:

```text
4 4
```

not:

```text
8
```

because the first two `2`s merge:

```text
2 2 → 4
```

and the second two merge:

```text
2 2 → 4
```

The newly-created `4` cannot immediately merge with another tile during the same move.

That's why after merging two tiles, you advance by two positions.

---

# 9. Implementing all four directions

You don't need four completely independent algorithms.

Implement **left** once.

Then transform the board for the other directions.

For example:

### Left

```text
→
```

Apply `merge_left()` to each row.

### Right

Reverse every row:

```text
[2, 0, 2, 4]

reverse

[4, 2, 0, 2]
```

Apply left merge:

```text
[4, 4, 0, 0]
```

Reverse again:

```text
[0, 0, 4, 4]
```

### Up

Transpose the matrix.

### Down

Transpose + reverse.

This drastically reduces the amount of game logic you have to maintain.

---

# 10. Detecting whether a move actually changed the board

This is extremely important for RL.

Suppose:

```text
2 4 8 16
0 0 0 0
0 0 0 0
0 0 0 0
```

and the agent chooses `UP`.

Nothing changes.

In actual 2048, an invalid/non-moving action should generally **not spawn a new tile**.

So:

```python
old_board = self.board.copy()

new_board, reward = move(old_board, action)

changed = not np.array_equal(old_board, new_board)
```

If:

```python
changed == False
```

then you shouldn't spawn a tile.

This distinction becomes very important during RL training.

---

# 11. The `step()` function

Your environment's core function becomes something like:

```python
def step(self, action):

    old_board = self.board.copy()

    new_board, merge_reward = move(self.board, action)

    changed = not np.array_equal(
        old_board,
        new_board
    )

    if not changed:
        return self.get_state(), 0, self.is_game_over(), {}

    self.board = new_board

    self.spawn_tile()

    self.score += merge_reward

    done = self.is_game_over()

    return (
        self.get_state(),
        merge_reward,
        done,
        {}
    )
```

This creates the RL transition:

$$
(s_t,a_t,r_t,s_{t+1})
$$

---

# 12. Terminal-state detection

A game ends when:

1. There are no empty cells.
2. No adjacent tiles can merge.

Suppose:

```text
2 4 8 16
32 64 128 256
512 1024 2 4
8 16 32 64
```

There are no empty cells.

You then need to check neighboring cells.

For each cell:

```text
right neighbor
down neighbor
```

If any pair is equal, the game isn't over.

Otherwise:

```python
done = True
```

You don't need to check every direction because checking right/down covers every neighboring pair exactly once.

---

# 13. State representation

This is one of the most important RL design decisions.

You *could* represent the board directly:

```text
[0, 0, 2, 0,
 0, 4, 0, 0,
 2, 2, 4, 0,
 0, 0, 0, 0]
```

But raw tile values have a major problem.

2048 values grow exponentially:

```text
2
4
8
16
32
64
...
2048
4096
8192
...
```

A neural network generally benefits from a more normalized representation.

---

# 14. Logarithmic encoding

Because every tile is a power of two, represent:

$$
2^n \rightarrow n
$$

For example:

```text
0     → 0
2     → 1
4     → 2
8     → 3
16    → 4
32    → 5
64    → 6
128   → 7
256   → 8
512   → 9
1024  → 10
2048  → 11
```

Mathematically:

$$
x' =
\begin{cases}
0 & x=0\\
\log_2(x) & x>0
\end{cases}
$$

This makes the numerical range much more manageable.

---

# 15. Even better: normalized logarithmic state

You can normalize the exponent:

$$
x' = \frac{\log_2(x)}{K}
$$

where \(K\) might be something like 16 or 20.

However, normalization isn't always necessary.

A very good starting state representation is simply:

```python
np.log2(board, where=board > 0)
```

with zeros retained as zero.

---

# 16. Alternative: one-hot representation

Another approach is to represent each tile using channels.

For example:

```text
channel 0 → empty
channel 1 → 2
channel 2 → 4
channel 3 → 8
...
channel N → 2^N
```

The state becomes approximately:

$$
N \times 4 \times 4
$$

rather than:

$$
4\times4
$$

This can work particularly well with CNNs.

For example:

```text
input shape = (16, 4, 4)
```

where each channel represents a tile exponent.

This is similar to how categorical board-game states are often represented.

---

# 17. Which state representation should you choose?

For a first implementation:

**Use logarithmic encoding.**

For example:

```text
0  0  1  0
0  2  0  0
1  1  2  0
0  0  0  0
```

instead of:

```text
0  0  2  0
0  4  0  0
2  2  4  0
0  0  0  0
```

It makes learning considerably easier.

---

# 18. Designing the reward

This is where RL becomes particularly interesting.

The obvious reward is:

$$
r_t = \text{merge score}
$$

For example:

```text
2 + 2 → 4
```

gives:

$$
r=4
$$

and:

```text
4 + 4 → 8
```

gives:

$$
r=8
$$

This is essentially the standard 2048 scoring system.

It's a good initial reward.

---

# 19. But reward design is a major research question

You can experiment with:

### Standard reward

$$
r = \text{merged tile value}
$$

### Log reward

$$
r = \log_2(\text{merged tile})
$$

### Number-of-merges reward

$$
r = \#\text{merges}
$$

### Survival reward

```text
+1 every valid move
```

### Game-over penalty

```text
-1 when game ends
```

### Maximum-tile reward

Potentially:

$$
r = \Delta \log_2(\text{max tile})
$$

But be careful.

**Reward shaping can fundamentally change what the agent learns.**

If you give too much reward for making small merges, the agent might learn to maximize immediate merges rather than create long-term board structure.

---

# 20. A sensible initial reward

I'd begin with:

$$
r_t =
\begin{cases}
\text{merge score} & \text{valid move}\\
0 & \text{invalid move}
\end{cases}
$$

and optionally:

```text
game over → -something
```

Then first establish that the agent can learn anything at all.

Only afterward experiment with reward shaping.

---

# 21. The Markov Decision Process

Now you can formally describe your 2048 environment as an MDP:

$$
M=(S,A,P,R,\gamma)
$$

where:

### State space

$$
S = \{4\times4 \text{ boards}\}
$$

### Action space

$$
A=\{UP,DOWN,LEFT,RIGHT\}
$$

### Transition function

$$
P(s'|s,a)
$$

The movement itself is deterministic, but tile spawning introduces stochasticity.

### Reward

$$
R(s,a,s')
$$

is your merge reward.

### Discount factor

$$
\gamma
$$

controls how much the agent values future rewards.

For 2048, you'd generally want a high discount factor, such as:

$$
\gamma \approx 0.99
$$

because creating a good board position now may pay off many moves later.

---

# 22. The stochastic component

An important conceptual point:

**2048 isn't deterministic from the agent's perspective.**

The agent chooses:

```text
LEFT
```

and the board movement is deterministic.

But then the environment randomly adds:

```text
2
```

or:

```text
4
```

to an empty location.

Therefore:

$$
P(s_{t+1}|s_t,a_t)
$$

is stochastic.

This makes 2048 a useful RL problem because the agent must learn a policy that performs well under uncertain future tile placements.

---

# 23. Choosing an RL algorithm

There are several possibilities.

For learning from scratch, I'd consider these in roughly this order:

### Option 1 — DQN

Deep Q-Network.

Learn:

$$
Q(s,a)
$$

where \(Q\) estimates the expected future return from taking action \(a\) in state \(s\).

The network outputs:

```text
UP       → Q(s,UP)
RIGHT    → Q(s,RIGHT)
DOWN     → Q(s,DOWN)
LEFT     → Q(s,LEFT)
```

This is conceptually straightforward.

---

# 24. DQN equation

The Bellman target is:

$$
y =
r+\gamma \max_{a'}Q_{\text{target}}(s',a')
$$

and the loss is:

$$
L =
\left(
y-Q_{\text{online}}(s,a)
\right)^2
$$

or often Huber loss instead of MSE.

The network learns to make:

$$
Q(s,a)
$$

more accurately approximate future return.

---

# 25. Why vanilla DQN isn't necessarily ideal for 2048

2048 has some difficult properties:

* enormous state space
* sparse-ish long-term objectives
* long episodes
* stochastic tile placement
* highly correlated experiences
* actions can be invalid
* reward depends heavily on long-term planning

So vanilla DQN may work, but you can get substantially better behavior with improvements.

A good progression is:

```text
DQN
 ↓
Double DQN
 ↓
Dueling DQN
 ↓
Prioritized Experience Replay
 ↓
n-step returns
```

You don't need all of them immediately.

---

# 26. Double DQN

Standard DQN uses:

$$
\max_a Q_{\text{target}}(s',a)
$$

which can produce overestimation.

Double DQN separates:

**action selection**

from:

**action evaluation**

using:

$$
a^* =
\arg\max_a Q_{\text{online}}(s',a)
$$

then:

$$
y =
r+
\gamma
Q_{\text{target}}(s',a^*)
$$

This is often a useful improvement.

---

# 27. Dueling architecture

Dueling DQN decomposes the value into:

$$
Q(s,a)
=
V(s)+A(s,a)
$$

where:

* \(V(s)\) = how good the board is
* \(A(s,a)\) = advantage of a particular action

This can make intuitive sense for 2048.

For some states, the board may be good regardless of whether you move left or right.

The network can learn:

```text
Board value
       +
Action-specific advantage
```

rather than learning every Q-value completely independently.

---

# 28. Experience replay

You don't want to train exclusively on consecutive experiences:

```text
s1 → s2 → s3 → s4 → ...
```

because they're highly correlated.

Instead store transitions:

$$
(s,a,r,s',done)
$$

in a replay buffer.

For example:

```python
replay_buffer.append(
    state,
    action,
    reward,
    next_state,
    done
)
```

Then randomly sample:

```python
batch = replay_buffer.sample(64)
```

and train on that batch.

This is one of the defining ideas of DQN.

---

# 29. Replay buffer

A simple implementation could use:

```python
from collections import deque

buffer = deque(maxlen=500_000)
```

Each experience:

```python
(
    state,
    action,
    reward,
    next_state,
    done
)
```

Then:

```python
batch = random.sample(buffer, batch_size)
```

For a serious implementation, you'll eventually want a more memory-efficient structure because storing millions of NumPy arrays can consume significant RAM.

---

# 30. Epsilon-greedy exploration

Initially, your agent knows nothing.

So it needs to explore.

A classic approach is:

$$
a =
\begin{cases}
\text{random action} & p=\epsilon\\
\arg\max_a Q(s,a) & p=1-\epsilon
\end{cases}
$$

Start with:

```text
epsilon = 1.0
```

meaning almost entirely random.

Then gradually reduce it:

```text
1.00
0.95
0.90
...
0.10
```

Eventually:

```text
epsilon ≈ 0.05
```

or lower.

---

# 31. But invalid actions create a problem

Suppose the board is:

```text
2 4 8 16
32 64 128 256
...
```

and UP does nothing.

If your agent repeatedly selects invalid actions, training can become wasteful.

You have two choices.

### Method A — Allow invalid actions

The agent learns that they are bad.

### Method B — Mask invalid actions

Before selecting an action, identify:

```python
valid_actions = env.valid_actions()
```

and only allow those.

For an initial DQN implementation, I'd actually recommend **action masking** during exploitation, because it avoids wasting decisions on moves that don't change the state.

However, be consistent between action selection and target calculation.

---

# 32. Neural network architecture

You don't need a huge network.

A simple fully connected model:

```text
4×4 board
   ↓
16 inputs
   ↓
Dense 256
   ↓
ReLU
   ↓
Dense 256
   ↓
ReLU
   ↓
Dense 4
```

produces:

```text
Q_up
Q_right
Q_down
Q_left
```

This is an excellent baseline.

---

# 33. Why CNNs can make sense

The board has spatial structure.

For example:

```text
2 4 8 16
...
```

has relationships between neighboring cells.

A convolutional network can learn local patterns such as:

* adjacent equal tiles
* empty spaces
* increasing sequences
* corners
* clusters
* monotonic rows/columns

So another architecture is:

```text
4×4×1
 ↓
Conv2D
 ↓
Conv2D
 ↓
Flatten
 ↓
Dense
 ↓
4 Q-values
```

I'd test both.

---

# 34. A particularly useful representation for CNNs

Instead of giving the CNN raw tile values, use exponent channels.

For example:

```text
channel 0 → tiles of value 2
channel 1 → tiles of value 4
channel 2 → tiles of value 8
...
```

Then:

$$
input \in \mathbb{R}^{N\times4\times4}
$$

where \(N\) is the maximum exponent you support.

This allows convolutional filters to recognize patterns associated with specific tile magnitudes.

---

# 35. The training loop

The basic DQN loop becomes:

```python
for episode in range(num_episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.select_action(state)

        next_state, reward, done, info = \
            env.step(action)

        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        if len(replay_buffer) >= batch_size:
            agent.learn()
```

That's the basic skeleton.

---

# 36. What `learn()` actually does

Suppose you sample:

```text
states
actions
rewards
next_states
dones
```

Calculate:

$$
Q(s,a)
$$

for the actions actually taken.

Then calculate target:

$$
y =
r+\gamma(1-done)
\max_{a'}Q_{\text{target}}(s',a')
$$

Then minimize:

$$
L = Huber(Q(s,a),y)
$$

and perform an optimizer step.

Conceptually:

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

# 37. Target network

A major DQN trick is to maintain two networks:

```text
Online network
Target network
```

The online network gets updated every training step.

The target network is updated much less frequently.

For example:

```python
target_network.load_state_dict(
    online_network.state_dict()
)
```

every 10,000 steps.

This prevents the target from changing too rapidly while you're trying to learn it.

---

# 38. Why this matters

Without a target network, your target is effectively:

$$
r+\gamma\max Q(s',a')
$$

while the same network is simultaneously changing \(Q\).

You're essentially trying to hit a moving target with another moving target.

The target network stabilizes learning.

---

# 39. Episode structure

One episode is:

```text
reset
 ↓
spawn 2 tiles
 ↓
agent chooses move
 ↓
move board
 ↓
spawn tile
 ↓
reward
 ↓
repeat
 ↓
game over
```

Then:

```text
new episode
```

You should track:

```python
episode_score
max_tile
episode_length
```

These are extremely useful metrics.

---

# 40. Don't use only average reward

For 2048, average reward can be misleading.

Track at least:

### Score

$$
\text{score}
$$

### Maximum tile

```text
max(board)
```

### Episode length

```text
number of valid moves
```

### Reached 128?

```text
percentage
```

### Reached 256?

```text
percentage
```

### Reached 512?

```text
percentage
```

### Reached 1024?

```text
percentage
```

### Reached 2048?

```text
percentage
```

This gives you a much better understanding of progress.

---

# 41. Evaluation must be separated from training

This is another common mistake.

During training:

```text
epsilon > 0
```

so the agent explores.

During evaluation:

```text
epsilon = 0
```

or near zero.

Then run, for example:

```text
1000 games
```

and report:

```text
mean score
median score
max score
mean max tile
2048 success rate
```

Otherwise exploration noise contaminates your evaluation.

---

# 42. Reproducibility

Your environment should support random seeds.

For example:

```python
env = Game2048(seed=42)
```

Use controlled RNGs rather than mixing:

```python
random
numpy.random
torch.random
```

without managing their seeds.

A reproducible experiment should be able to say:

```text
seed = 42
algorithm = Double DQN
learning_rate = ...
gamma = ...
architecture = ...
```

and reproduce approximately the same result.

---

# 43. Unit-test the environment before training

This is **absolutely critical**.

Don't start training until the game engine passes tests.

Test:

```text
[2,2,0,0] → [4,0,0,0]
```

```text
[2,2,2,0] → [4,2,0,0]
```

```text
[2,2,2,2] → [4,4,0,0]
```

```text
[4,4,4,4] → [8,8,0,0]
```

```text
[2,0,2,4] → [4,4,0,0]
```

and so forth.

These tests catch subtle errors that would otherwise make RL learning nearly impossible.

---

# 44. Test rotational equivalence

Because 2048 should be symmetric under board rotation/reflection, test that:

```text
LEFT
```

and:

```text
rotate board
UP
rotate back
```

produce equivalent results.

This is an excellent way to verify your movement implementation.

---

# 45. Separate game logic from RL logic

Your project structure could look like:

```text
2048-rl/
│
├── environment/
│   ├── game.py
│   ├── board.py
│   └── tests/
│
├── agent/
│   ├── dqn.py
│   ├── network.py
│   └── replay_buffer.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── configs/
│   └── dqn.yaml
│
├── checkpoints/
│
└── logs/
```

This makes experimentation much easier.

---

# 46. A useful class architecture

I'd structure it roughly like:

```python
class Board:
    def move(self, direction):
        ...

    def spawn_tile(self):
        ...

    def valid_moves(self):
        ...

    def is_game_over(self):
        ...


class Game2048:
    def reset(self):
        ...

    def step(self, action):
        ...

    def get_state(self):
        ...

    def render(self):
        ...


class ReplayBuffer:
    def add(self, transition):
        ...

    def sample(self, batch_size):
        ...


class DQN:
    def forward(self, state):
        ...

    def select_action(self, state):
        ...

    def update(self, batch):
        ...


class Trainer:
    def train(self):
        ...

    def evaluate(self):
        ...
```

This separation is worth doing from the beginning.

---

# 47. A crucial improvement: valid-action handling

You can have the environment expose:

```python
def valid_actions(self):
    valid = []

    for action in range(4):

        new_board, _ = move(self.board, action)

        if not np.array_equal(
            self.board,
            new_board
        ):
            valid.append(action)

    return valid
```

Then the agent can choose only from those actions.

For DQN, you can also mask invalid actions when computing:

$$
\max_a Q(s,a)
$$

by setting invalid actions to a very negative number.

Conceptually:

```python
q_values[invalid_actions] = -inf
```

This can dramatically improve sample efficiency.

---

# 48. But don't accidentally leak information

Action masking is acceptable because the environment can legitimately tell the agent which actions are valid.

However, think carefully about what information your observation provides.

The agent should receive information that a player could reasonably derive from the current board.

For example:

```text
board
valid actions
```

is fine.

Giving:

```text
future spawned tile
```

would obviously break the problem.

---

# 49. The biggest challenge: long-term strategy

2048 isn't just about immediate merges.

Consider:

```text
2 4 8 16
2 4 8 16
...
```

The agent needs to learn structural concepts such as:

* keeping large tiles in a corner
* maintaining monotonic rows
* preserving empty cells
* avoiding isolated high-value tiles
* building merge chains
* avoiding destructive movements

This is why purely immediate reward isn't enough.

The value function needs to estimate:

$$
\text{future potential}
$$

rather than merely:

$$
\text{immediate merge score}
$$

---

# 50. Hand-designed heuristics can help you understand the problem

Before RL, implement a heuristic player.

For example:

$$
H(s)
=
w_1 M(s)
+
w_2 E(s)
+
w_3 C(s)
+
w_4 S(s)
$$

where:

* \(M(s)\) = monotonicity
* \(E(s)\) = number of empty cells
* \(C(s)\) = corner/highest-tile quality
* \(S(s)\) = smoothness

This isn't necessarily the final RL solution.

But it gives you a **baseline**.

---

# 51. Why the heuristic baseline matters

Suppose your RL agent reaches:

```text
maximum tile = 64
```

You don't know whether that's good.

But if a simple heuristic reaches:

```text
256
```

you know your RL implementation needs work.

You should ideally compare:

```text
Random
↓
Greedy
↓
Heuristic
↓
DQN
↓
Double DQN
↓
Dueling Double DQN
```

This makes your experiments scientifically meaningful.

---

# 52. Random baseline

A random agent:

```python
action = random.choice(valid_actions)
```

should be your first baseline.

It answers:

> Does the environment itself behave plausibly?

If random play produces bizarrely high scores, your environment may have a bug.

---

# 53. Greedy baseline

A simple greedy agent chooses the action with the largest immediate reward:

$$
a=\arg\max_a r(s,a)
$$

This usually isn't particularly strong.

That's actually useful.

It demonstrates why long-term value matters.

---

# 54. Heuristic baseline

A stronger baseline evaluates the resulting board.

For example:

$$
H(s)=
\alpha E(s)
+
\beta M(s)
-
\gamma P(s)
$$

where:

* \(E\) = empty cells
* \(M\) = monotonicity
* \(P\) = penalties for undesirable tile arrangements

Then:

```python
action = argmax(H(next_state))
```

This can become surprisingly strong.

---

# 55. Advanced approach: expectimax

Interestingly, 2048 has historically been approached very effectively using **expectimax search**.

Instead of treating the environment simply as an RL problem, you can model:

```text
Player chooses move
       ↓
Random tile appears
       ↓
Player chooses move
       ↓
Random tile appears
...
```

The tree alternates between:

```text
MAX node
```

and:

```text
CHANCE node
```

The expected value is:

$$
V(s)
=
\max_a
\sum_{s'}
P(s'|s,a)V(s')
$$

This is computationally expensive but makes a powerful benchmark.

---

# 56. RL versus search

This gives you an interesting experimental question:

> Can a neural RL agent learn a policy that approaches the performance of search-based 2048 players?

That's a much more interesting research project than simply "train DQN on 2048."

You can compare:

```text
Random
Greedy
Expectimax
DQN
Double DQN
Dueling DQN
```

---

# 57. Curriculum learning

Another possibility is to make the environment easier initially.

For example:

```text
Stage 1:
3×3 board
```

then:

```text
Stage 2:
4×4 board
```

Or modify tile spawning.

However, I'd recommend first solving standard 4×4 2048.

Otherwise you're adding another research variable before knowing whether the basic implementation works.

---

# 58. Customizing 2048

Once standard 2048 works, you can modify the environment.

For example:

### Different board sizes

```text
3×3
4×4
5×5
```

### Different spawn probabilities

```text
2: 80%
4: 20%
```

### More tile types

For example:

```text
2
4
8
16
...
```

with alternative spawning rules.

### Obstacles

```text
X 0 0 2
0 4 X 0
```

### Special tiles

For example:

```text
wildcard tile
blocked tile
bonus tile
```

### Different merge rules

For example:

```text
2 + 4 → 8
```

instead of standard equal-tile merging.

At that point, you've built a **generalized 2048 RL environment** rather than simply cloning the game.

---

# 59. Make the rules configurable

Instead of hard-coding everything:

```python
class GameConfig:

    board_size = 4

    spawn_values = [2, 4]

    spawn_probabilities = [0.9, 0.1]

    target_tile = 2048

    invalid_move_penalty = 0
```

Then:

```python
env = Game2048(config)
```

This lets you run experiments without rewriting the environment.

---

# 60. Configuration-driven experiments

For example:

```yaml
environment:
  board_size: 4
  target_tile: 2048
  spawn_probability_2: 0.9
  spawn_probability_4: 0.1

agent:
  algorithm: double_dqn
  hidden_size: 256

training:
  learning_rate: 0.0001
  gamma: 0.99
  batch_size: 128
  replay_size: 500000
```

Then you can change experiments systematically.

---

# 61. Logging

Every training run should record something like:

```text
episode
score
max_tile
episode_length
epsilon
loss
Q_mean
Q_max
```

Then plot:

```text
score vs episode
```

and:

```text
max tile vs episode
```

and perhaps:

```text
loss vs training step
```

But remember: a decreasing loss doesn't necessarily mean the agent is becoming better.

**Game performance is the ultimate metric.**

---

# 62. Checkpointing

Save the neural network periodically:

```text
checkpoint_100k.pt
checkpoint_500k.pt
checkpoint_1m.pt
```

Also save:

```text
optimizer state
epsilon
training step
configuration
random seed
```

Otherwise you may not be able to reproduce the exact experiment.

---

# 63. The complete development methodology

I'd build the project in this order:

### Phase 1 — Game engine

Implement:

```text
Board
 ↓
move
 ↓
merge
 ↓
spawn
 ↓
terminal detection
```

### Phase 2 — Testing

Test every movement and merge edge case.

### Phase 3 — RL interface

Implement:

```text
reset()
step()
state
reward
done
```

### Phase 4 — Baselines

Implement:

```text
random
greedy
heuristic
```

### Phase 5 — DQN

Implement:

```text
network
replay buffer
epsilon-greedy
target network
training
```

### Phase 6 — Evaluation

Run hundreds/thousands of evaluation games.

### Phase 7 — Improvements

Experiment with:

```text
Double DQN
Dueling DQN
Prioritized replay
n-step returns
CNN representation
action masking
```

### Phase 8 — Custom 2048

Only after everything works, modify the game rules.

---

# 64. What the final architecture looks like

The entire system can be thought of as:

```text
                  ┌───────────────┐
                  │    Board      │
                  │   4 × 4      │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ State Encoder │
                  │               │
                  │ log2 / onehot │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Neural Network│
                  │               │
                  │ Q(s,a)        │
                  └───────┬───────┘
                          │
                 ┌────────┴────────┐
                 │                 │
             exploration       exploitation
                 │                 │
                 └────────┬────────┘
                          ▼
                     action
                          │
                          ▼
                  ┌───────────────┐
                  │ 2048 Engine   │
                  │               │
                  │ move          │
                  │ merge         │
                  │ spawn         │
                  │ reward        │
                  └───────┬───────┘
                          │
                          ▼
                  next_state/reward
                          │
                          ▼
                  ┌───────────────┐
                  │ Replay Buffer │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ DQN Training  │
                  └───────────────┘
```

---

# 65. Recommended first implementation

If you're genuinely building this **from scratch**, I would deliberately avoid making the first version overly sophisticated.

Use:

| Component      | Initial choice        |
| -------------- | --------------------- |
| Board          | 4×4                   |
| Actions        | 4                     |
| State          | log2 tile values      |
| Reward         | merge score           |
| Algorithm      | DQN                   |
| Network        | 2×256 fully connected |
| Replay         | 100k–500k             |
| Batch          | 64–128                |
| Discount       | 0.99                  |
| Exploration    | epsilon-greedy        |
| Target network | yes                   |
| Optimizer      | Adam                  |
| Evaluation     | 500–1,000 games       |
| Baselines      | random + heuristic    |

Once that works, move to:

```text
Double DQN
        ↓
Dueling DQN
        ↓
Prioritized replay
        ↓
n-step returns
        ↓
CNN / richer representation
```

That progression makes it much easier to understand **which improvement actually matters**.

---

## One particularly important point

If your goal is to **learn RL rather than merely get a high-scoring 2048 bot**, don't start with a library that hides the environment and algorithm.

Build these pieces yourself:

```text
1. 2048 board mechanics
2. Environment API
3. State encoder
4. Random baseline
5. Replay buffer
6. Q-network
7. DQN update
8. Target network
9. Training loop
10. Evaluation framework
```

Then you have a complete understanding of the pipeline:

$$
\boxed{
\text{Game}
\rightarrow
\text{MDP}
\rightarrow
\text{Experience}
\rightarrow
\text{Q-function}
\rightarrow
\text{Policy}
\rightarrow
\text{Learning}
}
$$

That is the core methodology behind building a custom 2048 RL system from first principles.

If you're implementing this as an actual project, the next useful step would be to turn this methodology into a **complete Python/PyTorch implementation**, starting with the 2048 engine and unit tests, then adding DQN one component at a time.
