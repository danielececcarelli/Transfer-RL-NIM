# Q-Learning Reinforcement Learning  for the game of NIM

### Reinforcement Learning to solve the game of NIM (https://en.wikipedia.org/wiki/Nim)

The aim of this project is to train the computer to play against human in the game of NIM. In particular, the final aim of this project will be to try to "transfer" the learning for playing the game with different rows.

This work is inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542 and its implementation of the game of TicTacToe (Tris).

### Rules of NIM

"Nim is a mathematical game of strategy in which two players take turns removing (or "nimming") objects from distinct heaps or rows. On each turn, a player must remove at least one object, and may remove any number of objects provided they all come from the same heap or row. The goal of the game is either to take the last object."

For example, N_rows = 5, object = {1,2,3,4,5}

![image_nim](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/Example-game-of-Nim-1-2-3-4-5-Each-row-represents-a-heap-Players-choose-a-row-and.png)

In this initial implementation, I have decided to set the number of rows to 5, and the initial state of the game is:

**O**

**O O**

**O O O**

**O O O O**

**O O O O O**

The player that take away the last "o" win the game.

### States Representation

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

### Number of States

Soon (plot of number of states vs `number_rows`)

### Transfer the Learning from different number_rows games

In order to re-use the policy from games with a smaller number of rows, we can simply add zeros in front of the keys.

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

### Epsilon search

Epsilon_min = 0.05

![image_epsilon](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/epsilon-1.png)

### Results of training

Let's see the results for different number of rows = {3,4,5}. (num_games = 50'000)
In the game, p1 plays always as first player.

**3 rows**: 
![image_3](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/output_avg_reward_3-1.png)

As we can see, in the 3-rows game the first player is going to lose every match. We can prove it that this is true also by theory of combinatory game: with 3 rows and just 6 objects, the tree of moves is quite small. In the first games we can see that p1 has bigger rewards (probaly because it has the first move), but with the time p2 is definitely going to win every matches.


**4 rows**: 
![image_4](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/output_avg_reward_4-1.png)

Differently from the first case, in the 4-rows game the first player is going to win the matches.
This is basically true because, if the player p1 starts with a move that erase all the objects in the 4-th row, the player p2 comes in a situation of 3-rows games, and we know that starting from that position will make you lose the match. And of course this move is often the first move that player p1 will make. 


**5 rows**: 
![image_5](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/output_avg_reward_5-1.png)

And finally the most difficult case: with 5 rows, we have a lot of different states (n=132) and the problem start to become interesting. Also here the p1 is going to win.


An interesting question now could be: can we use the knowledge of games with smaller number of rows (3,4) to gain knowledge for the bigger cases (like 5 or more)? 
How much time (in terms of games played) can I save with this method?

### Can we transfer learning to game with bigger N_ROWS ?

As said before, we can use for example policy from `number_rows = 3` to speed up the learning of games with `number_rows = 5`. Here the result with just n_games = 25'000 (instead of 50'000 as in the previous case).

![image_5_from3](https://github.com/danielececcarelli/Q-Learning-RL-Game-Of-Nim/blob/master/images/output_avg_reward_5_from3-1.png)


