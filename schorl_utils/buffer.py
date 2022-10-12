import random
import numpy as np

from collections import namedtuple, deque
from queue import PriorityQueue


replaybatch = namedtuple("replaybatch", ['states', 'actions', 'rewards', 'next_states', 'dones'])
replayitem = namedtuple("replayitem", ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, maxsize:int = None) -> None:
        ''' Initialize the ReplayBuffer
        Args:
            maxsize (int): size of replay buffer
        '''
        self.buffer = deque(maxlen=maxsize)

    def put(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batchsize:int = 1) -> replaybatch:
        ''' Sample the ReplayBuffer with batchsize
        Args:
            batchsize (int): default is 1
        Return:
            replaybatch (replaybatch): states:np.array, 
        '''
        # try:    
        sample_batch = random.sample(self.buffer, batchsize)    # zip each item to batch
        states, actions, rewards, next_states, dones = zip(*sample_batch)
        return replaybatch(states=np.array(states), actions=actions, rewards=rewards, next_states=np.array(next_states), dones=dones)
        # except Exception as e:
            # print(e)
            # raise Exception("the buffer can't be sampled, because batchsize is larger than buffer's size")

    def __len__(self) -> int:
        return len(self.buffer)


"""
Compare collections.deque with queue.PriorityQueue

%%timeit
import queue

a = queue.PriorityQueue(maxsize=2000)

for i in range(5000):
    if a.full():
        a.get()
    a.put((i,i))

---
15.4 ms ± 155 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%%timeit
from collections import deque

a = deque(maxlen=2000)

for i in range(5000):
    a.append((i,i))

---
380 µs ± 1.63 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

"""