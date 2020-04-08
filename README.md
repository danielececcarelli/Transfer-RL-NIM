# Transfer in Reinforcement Learning  for the game of NIM

#### Author : Daniele Ceccarelli

## Reinforcement Learning to solve the game of NIM 

(https://en.wikipedia.org/wiki/Nim)

The aim of this project is to train the computer to play against human in the game of NIM. In particular, the final aim of this project will be to try to "transfer" the learning for playing the game with different rows.

This work is inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542 and its implementation of the game of TicTacToe (Tris).

### Rules of NIM

"Nim is a mathematical game of strategy in which two players take turns removing (or "nimming") objects from distinct heaps or rows. On each turn, a player must remove at least one object, and may remove any number of objects provided they all come from the same heap or row. The goal of the game is either to take the last object."

For example, N_rows = 5, object = {1,2,3,4,5}

![image_nim](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/Example-game-of-Nim-1-2-3-4-5-Each-row-represents-a-heap-Players-choose-a-row-and.png)

In this initial implementation, I have decided to set the number of rows to 5, and the initial state of the game is:

**O**

**O O**

**O O O**

**O O O O**

**O O O O O**

The player that take away the last "o" win the game.

# States Representation

### Board and Moves

Boards of game are represented in our algorithm as an array of length `number_rows` and is initially defined as:
```python
board = np.arange(1,number_rows+1)
```
So, the initial state (in case of, for example, `number_rows = 5`) is
`board = [1,2,3,4,5]`

In each move we remove objects from a row (from 1 object up to the current value of a row) and then we do a resort:
```python
def updateState(self, move):
        row = move[0] # from which row we are going to remove objects
        num = move[1] # number of objects to remove
        # remove num objects from the correct row
        self.board[row] = self.board[row]-num
        
        # resort the board to try to keep small the different states
        self.board = np.sort(self.board)
```

### Why should we need a sorting after each move? 

This is done to try to keep as small as possible the number of total states: this is a crucial point in the Q-Learning algorithm, that works better with a small number of states.

Let's see an example with `number_rows = 2`. We have these states:
- [0,0]
- [0,1]
- [1,0]
- [1,1]
- [2,0]
- [0,2]
- [1,2]
- [2,1]

but of course some states are equivalent (mirror states) like [0,1] or [1,0]. To keep just one of these two states,
we can use only the one that is sorted ([0,1]); in this way we have just :
- [0,0]
- [0,1]
- [1,1]
- [0,2]
- [1,2]

5 states against 8 without sorting. And this difference increase with `number_rows`.
With `number_rows = 3` we can manage to keep just 14 states against 24 without this technique. 

### Number of Different States

| `number_rows`  | Number different states |
| ------------- | ------------- |
| 1 | 2 |
| 2 | 5 |
| 3 | 14 |
| 4 | 42 |
| 5 | 132 |
| 6 | 429 |
| 7 | 1430 |

Plot of number of different states vs `number_rows`

![image_state](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/plot_states.png)

As we can see, the curve follow a clear exponential growth.

# Exploring-Exploiting tradeoff

### Epsilon search 

How do we decide when it is time to explore new states (exploring) and when it is time to use 
our best strategy so far (exploiting) ?

We use a simple Epsilon search algorithm: at every iteration (or in other words, at every move in a game, 
and at every game in a range of n_rounds) sample u from a Uniform(0,1), and :

- if u > epsilon(game, parameters) 
        -> EXPLOITING
        
- else
        -> EXPLORING
        
where epsilon has an exponential decreasing with respect to the game in the range on n_rounds with this parameter:

#### Epsilon = max(epsilon_min, (exp_rate * exp(-episode * decay_rate)))

where :
- episode = number of that game / number of total games to be play (goes from 0 to 1)
- epsilon_min = 0.05
- exp_rate = 0.99
- decay_rate = 5

![image_epsilon](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/epsilon-1.png)


# Q-learning algorithm

### States-Value Dictionary

To collect the pair states-value, we use a python dict in the Player class :
`self.states_value = {}`.

In this dict we use as key the position of the current board (for example '[0,0,0]') transformed to string and the value associated is the current value in terms of reward of that particular state (in this example the value will be >0 because if you left the board with 
the state [0,0,0] you have win the game).

### Reward and Update state

If the agent win the game, he will be rewarded with a +1, otherwise -1; we collect the history of all the games and who win
the `self.win_games = []` for each of the two player that are play (p1 and p2): after the learning phase, p1.win_games and p2.win_games
will be a sequence of +1 or -1, and of course if p1.win_games[i]=+1 (so p1 win the i-th game), p2.win_games[i]=-1.

But how can we use the information of player who win the game to update the current value of a state in `self.states_value` during learning?

In this type of RL we can reward all the states used to win a game only after the end of the game using the Bellmann eq; for this reason we use FeedReward methon:

```python
# at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        initial_reward = reward
        for st in reversed(self.states): # all states used in that game
            if self.states_value.get(st) is None: # if this is the first time we have used that state
                self.states_value[st] = 0
            # Bellmann Equation    
            self.states_value[st] += self.lr * (initial_reward + self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]
```


# Transfer the Learning from different number_rows games

In order to re-use the policy from games with a smaller number of rows, we can simply add zeros in front of the keys.

for example, if we have in our dict `states_value` the pair key-value of states_value('[0,2,2]') = 0.9 from a previous learning 
experience for the case of `number_rows = 3`, we can reuse it in case of `number_rows = 4` by just setting :

states_value('[0,0,2,2]') = 0.9

```python
# load the policy to play against human    
def loadPolicy_from_previous_cases(self, file, n_previous_cases, n_rows):
        fr = open(file, 'rb')
        tmp = pickle.load(fr)
        fr.close()
        zeros_add = n_rows - n_previous_cases

        tmp_copy = tmp.copy()

        zeros_string = ''
        for i in range(zeros_add):
            if i == 0:
                zeros_string += '[0 '
            else:
                zeros_string += '0 '

        for k, v in tmp_copy.items():
            new_key = zeros_string + k[1:]
            tmp[new_key] = tmp.pop(k)

        self.states_value = tmp
```

# Results of training

Let's see the results for different number of rows = {3,4,5,6}. (num_games = 50'000)
In the game, p1 plays always as first player.

We plot the mean reward of the previous 1000 games. We recall that if the agent win, it gets as reward +1; 
while it gets a -1 if it lose the game.

**3 rows**:

![image_3](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/output_avg_reward_3_10000_games.png)

As we can see, in the 3-rows game the first player is going to lose every match. We can prove it that this is true also by theory of combinatory game: with 3 rows and just 6 objects, the tree of moves is quite small. 

**4 rows**: 

![image_4](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/output_avg_reward_4_10000_games.png)

Differently from the first case, in the 4-rows game the first player is going to win the matches.
This is basically true because, if the player p1 starts with a move that erase all the objects in the 4-th row, the player p2 comes in a situation of 3-rows games, and we know that starting from that position will make you lose the match. And of course this move is often the first move that player p1 will make. 


**5 rows**: 

![image_5](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/output_avg_reward_5_10000_games.png)

And finally the most difficult cases: with 5 rows (and of course also with 6 rows), we have a lot of different states (n=132) and the problem start to become interesting. Also here the p1 is going to win.

**6 rows**: 

![image_6](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/output_avg_reward_6_10000_games.png)

As in the case of 5 rows, with 6 rows and 429 different states to visit the problem become very interesting and challenging.
The choice of epsilon-search parameters now is very important: we have to balance the trade-off of exploring new states 
(that are a lot!) and exploting the best strategy.

An interesting question now could be: can we use the knowledge of games with smaller number of rows (3,4 or 5) to gain knowledge for the bigger cases (like 6 or more)? 
How much time (in terms of games played) can I save with this method? Is the learning curve faster with this method? 

### Can we transfer learning to game with bigger N_ROWS ?

As said before, we can use for example policy from `number_rows = 5` to speed up the learning of games with `number_rows = 6`. 

![image_6_from5](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/output_avg_reward_6_10000_games_from5.png)

If we compare this plot with the previous of 6 rows we don't see much difference, probably the gain is not so big. Let's try to analyze
in the detail the speed-up that we can have using the learning of previous cases.

What can we expect as "improvement" of transfer learning?

In general, these are the three improvement that we can have:

![image_improv](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/The-three-objectives-of-Transfer-Learning.png)

(Image from Genevay, Aude & Laroche, Romain. (2016). Transfer Learning for User Adaptation in Spoken Dialogue Systems)

In this particular example (NIM) we can only expect a jumpstart impr. or a learning speed impr., not
a asymptotic impr. because we expect that our agent reach always the maximum he can afford, that is
the knowledge of the perfect strategy.

# Result of transfer learning

## First approach

### Case of 6 rows using knowledge from 5 rows

Let's analyze 5 sample of the same experiment (10000 rounds, learn to play 6 from 5, epsilon_min = 0.01).
In the following images, we plot the % of winning games in the previous 2000 rounds, setting the reward = 1 if the agent 
with that games, 0 otherwise. In the last part of the training we can expect that both the agent tends to the upperbound of 1,
when the agent wins all the previous 2000 games.

![prima_prova](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_primaprova.png)

![seconda_prova](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_secondaprova.png)

![terza_prova](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_terzaprova.png)

![quarta_prova](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_quartaprova.png)

![quinta_prova](https://github.com/danielececcarelli/Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_quintaprova.png)

As we can see, the results seem to be quite noisy, we don't see always a clear improvement of the learning curve "6_from_5" with respect
to the classical "6". In the first 4 images we can see an improvement in using the knowledge of 5_rows case, while this is not
true in the last plot. 

We need to investigate this deeply; probably a good idea could be change the way we learn when we have the knowledge of smaller cases:
let's analyze as example a case where we want to learn how to play with 3 rows using also the knowledge of 2 rows.
We have all the state that we have seen before : 
- [0,0]
- [0,1]
- [1,1]
- [0,2]
- [1,2]

that now, in terms of states of 3_rows framework, becomes :
- [0,0,0]
- [0,0,1]
- [0,1,1]
- [0,0,2]
- [0,1,2]

If now, in one of our game, we reach these state, we don't need no more to learn how to move from this, so we don't
need the "exploration" part of our epsilon-search algorithm, because we already know the best strategy of these states, and
if we enter in one of these states, we stay up to the end of the game in this set of states (the states of 2_rows game).

Of course this approach has some problem: 
for example if our current state is [1,2,2] we can go in one the states of 2_rows game (in particular [0,1,2] ) but laso in
other states, like [1,1,2] or [0,2,2], that don't belong to the set of states of 2_rows game. Which move should I take?
This of course depends on the current value associated to each of the three states, but should we trust more in the value of
[0,1,2] that we inherit from 2_rows learning respect to the other 2, where we are currently learning, especially when we are in the
previous step of the games? 

## Second Approach

### version 2.0

In the new version of the algorithm (v2), we add a new member function in State class to check if a state of N rows
belong also to a lower n rows cases.

```python
# check if the state is also from n_previous game
def is_previous_state(self, n_previous):
        skip = self.number_rows - n_previous -1
        # self.board is already sorted
        for i in range(n_previous+1):
         if (self.board[i+skip]>i):
             return 0
        return 1
```

In this new approach to Transfer Learning in 6 rows game (with transfer learning from 5 rows), we choose to use a smaller 
epsilon (for the epsilon-search algorithm) when we reach a state from 5-rows game, so when the is_previous_state(5)
is True. We set a parameter for the division of the original epsilon to 3.

Let's look at an example. We are learning with 10000 rounds, using the knowledge of 5-rows game to 
gain knowledge for 6-rows game. 
We are in the first steps of the "for loop", where epsilon is still big (for
instance epsilon(i) = 0.9). When we reach in the "i-th" game a state belonging to 5-rows game (so, when 
is_previous_state(5) is True), we change epsilon(i) with epsilon(i)/3 = 0.3

This is the new definition of chooseAction:
```python
def chooseAction(self, moves, current_board, symbol, episode, divide = 1):
        # if our random unif > epsilon
        if np.random.uniform(0, 1) >= max(self.epsilon_min, (self.exp_rate * math.exp(-episode*self.decay_rate))/divide):
```
when ```divide = 1```, nothing change from the previous approach; while if ```divide = 3``` we are using the new approach.

The result of this second approach (approach (b) ) is bigger: we get always a learning speed improvement and sometimes 
also a jumpstart.

# Comparison between the two different approach of Transfer Learning
### 6: Learn 6-rows game from scratch
### 6_from5_a : first approach
### 6_from5_b : second approach

![prova_a](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_a.png)

![prova_b](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_b.png)

![prova_c](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_c.png)

![prova_d](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_d.png)

![prova_e](https://github.com/danielececcarelli/Transfer-Reinforcement-Learning-for-NIM-Game/blob/master/images/learn5_vs_6_e.png)
