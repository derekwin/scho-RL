"""
useful function for data process and data conversion
"""
from tqdm import tqdm
# from .decorators import tensorboard_decorator
from tensorboardX import SummaryWriter
from .buffer import replayitem
from .envs import get_device
import torch
import numpy as np
import enum

class Agent:
    """Agent Class
    Methods
        __init__  : 初始化通用参数
        __call__  : 执行get_action()
            agent = Agent()
            action = agent(state)
        get_action  : return action # 如果需要定制，可以复写该函数
            return action
        update    : learning and update
        save_net  : save net state_dict
    """
    def __init__(self, 
            device = get_device(),
            gamma = 0.9,
            epsilon = 0.1,
            lr = 2e-3,
            optim = torch.optim.Adam,
            loss = torch.nn.functional.mse_loss,
            data_type = torch.float,
            ) -> None:
        """
        基类初始化一些通用的默认参数
        """
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.optim =optim
        self.loss = loss
        self.data_type = data_type

    @classmethod
    def save_net(self, net, filepath):
        """ 
        保存网络到对应目录
        """
        torch.save(net.state_dict(), filepath)
    
    def get_action(self, *args):
        """
        返回决策的动作
        """
        raise NotImplementedError("You need to rewrite this method to return a action")

    def __call__(self, *args):
        return self.get_action(args)
    
    def update(self):
        """
        此处实现算法的更新逻辑

        在此函数返回要在tensor中自动记录的参数，如loss
        return loss_a, loss_b
        """
        raise NotImplementedError("You need to rewrite this method to learn")


class UPDATE_MODE(enum.Enum):
    SINGLE_STEP = 1
    MULTI_STEP = 2


class Train:
    """A Train framework integrated Tensorboard log function and tqdm bar
    
    Args:
        __init__ : init paramters
        run_episode : [rewrite this] logitic of each episode
        train : trainning main loop, integrated with tensorboard and tqdm

    """
    
    def __init__(self, 
        env, 
        agent, 
        tblogpath = './logs', 
        if_render = False,
        update_mode = UPDATE_MODE.SINGLE_STEP,
        device = get_device(),
        data_type = torch.float
        ) -> None:
        self.env = env
        self.agent = agent
        self.tblogpath = tblogpath
        self.if_render = if_render
        self.update_mode = update_mode
        self.device = device
        self.data_type = data_type

    def run_episode(self):
        """
        rewrite this function to achieve new env interact
        Default:
            default is dqn run in CartPole-v1
        Return:
            {'item', itemvalue}
        """
        raise NotImplementedError("You need to rewrite this method to run episode")
    
    def gymstate_to_tensor(self, data):
        return torch.tensor(data, dtype=self.data_type).to(device=self.device)

    def train(self, episodes:int, devide_episodes:int=10):
        tbwriter = SummaryWriter(self.tblogpath)
        for i in range(devide_episodes):
            with tqdm(total=int(episodes/devide_episodes), desc=f'Iteration {i}') as pbar:
                for episode in range(int(episodes/devide_episodes)):
                    # prepare data
                    if self.update_mode is UPDATE_MODE.MULTI_STEP:
                        obs_list = []
                        args_list = []

                    done = False
                    reward_list = []
                    loss_list = []
                    state = self.env.reset()
                    # convert state to tensor
                    # train with batch, so run with batch too
                    state = self.gymstate_to_tensor([state])
                    while not done:
                        # get action
                        action, *args = self.agent.get_action(state)  # 采样类函数会返回动作及其概率

                        # convert the action data to fit env, default is convert to np.ndarray
                        # action = action.detach().numpy()
                        # step env with action
                        next_state, reward, done, *info = self.env.step(action) 

                        # convert next_state, action, reward, done to tensor
                        each_step = replayitem(
                                    state = state, 
                                    action = self.gymstate_to_tensor(action), 
                                    reward = self.gymstate_to_tensor(reward),
                                    next_state = self.gymstate_to_tensor([next_state]),
                                    done=self.gymstate_to_tensor(done))
                        
                        # update if single step update mode
                        if self.update_mode is UPDATE_MODE.SINGLE_STEP:
                            loss = self.agent.update(each_step, *args)
                            loss_list.append(loss)
                        
                        reward_list.append(reward)
                        if self.update_mode is UPDATE_MODE.MULTI_STEP:
                            obs_list.append(each_step)
                            args_list.append(args)
                        state = each_step.next_state
                        
                    if self.update_mode is UPDATE_MODE.MULTI_STEP:
                        loss = self.agent.update(obs_list, args_list)
                        loss_list.append(loss)

                    pbar.set_postfix({
                        'episode': episodes / devide_episodes * i + episode + 1,
                        'mean reward': '%.3f' % np.mean(reward_list)
                    })
                    pbar.update(1)

        self.env.close()
        tbwriter.close()