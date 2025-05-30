import json
from MahjongGB import MahjongFanCalculator
import torch
import torch.optim as optim
from enum import Enum
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
from copy import deepcopy
from torch.distributions import Categorical
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
import argparse
import time
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class cards(Enum):
    # 饼万条
    B = 2
    W = 1
    T = 0
    # 风
    F = 3
    # 箭牌
    J = 4
class requests(Enum):
    BUHUA = 0
    DRAW = 4
    PLAY = 5
    PENG = 6
    CHI = 7
    GANG = 8
    BUGANG = 9
    MINGGANG = 10
    ANGANG = 11

class responses(Enum):
    PASS = 0
    PLAY = 1
    HU = 2
    # 需要区分明杠和暗杠
    MINGGANG = 3
    ANGANG = 4
    BUGANG = 5
    PENG = 6
    CHI = 7
    need_cards = [0, 1, 0, 0, 1, 1, 0, 1]
    loss_weight = [1, 1, 5, 2, 2, 2, 2, 2]



class Mymodel_action(nn.Module):
    def __init__(self):
        super(Mymodel_action, self).__init__()

        # ResNet block
        self.resnet = nn.Sequential(
            nn.Conv2d(117, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 1
        self.concat1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 2
        self.concat2 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Output block
        self.output = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 235)  # Replace 10 with the number of output classes
        )

    def mask_actions(actions, valid_actions):
        masked_actions = np.zeros_like(actions)
        for i, valid in enumerate(valid_actions):
            if valid:
                start = sum(valid_actions[:i])
                end = start + actions[start:end].shape[0]
                masked_actions[start:end] = actions[start:end]
        return masked_actions

    def forward(self, x,mask):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加批次维度
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)  # 添加批次维度
            
        # 调整通道顺序 (H, W, C) -> (C, H, W)
        # x = x.permute(0, 3, 1, 2)
        out_resnet = self.resnet(x)
        out_concat1 = self.concat1(out_resnet)
        out_concat2 = self.concat2(torch.cat([out_resnet, out_concat1], dim=1))
        out_output = self.output(torch.cat([out_concat2, out_concat1], dim=1))
        masked_output = out_output * mask  # 将非法动作的概率设为0
        return F.softmax(masked_output, dim=-1)


class Mymodel_value(nn.Module):
    def __init__(self):
        super(Mymodel_value, self).__init__()

        # ResNet block
        self.resnet = nn.Sequential(
            nn.Conv2d(117, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 1
        self.concat1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 2
        self.concat2 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Output block
        self.output = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加批次维度
        out_resnet = self.resnet(x)
        out_concat1 = self.concat1(out_resnet)
        out_concat2 = self.concat2(torch.cat([out_resnet, out_concat1], dim=1))
        out_output = self.output(torch.cat([out_concat2, out_concat1], dim=1))
        return out_output

class Mymodel_reward(nn.Module):
    def __init__(self):
        super(Mymodel_reward, self).__init__()

        # ResNet block
        self.resnet = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 1
        self.concat1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Concatenate block 2
        self.concat2 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Output block
        self.output = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 9 * 184, 1)  # Replace 10 with the number of output classes
        )

    def forward(self, x):
        out_resnet = self.resnet(x)
        out_concat1 = self.concat1(out_resnet)
        out_concat2 = self.concat2(torch.cat([out_resnet, out_concat1], dim=1))
        out_output = self.output(torch.cat([out_concat2, out_concat1], dim=1))
        return out_output


#记录对局数据
class MahjonggMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.dones = []

    def add(self, state, action, logprobs,reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprobs)
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.logprobs.pop(0)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []

    def get_batches(self, batch_size):
        indices = np.random.permutation(len(self.states))
        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        return [(np.array(self.states)[batch], np.array(self.actions)[batch],
                 np.array(self.rewards)[batch], np.array(self.dones)[batch]) for batch in batches]

from tqdm import tqdm
import time


# 定义Off-policy PPO算法
class OffPolicyPPO:
    def __init__(self,  lr=3e-4, gamma=0.99, clip_ratio=0.2, entropy_coef=0.01):
        self.policy = Mymodel_action()
        self.value=Mymodel_value()
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

    def train_step(self, states, actions, old_log_probs, returns, advantages, masks):
        # 计算策略函数的新的概率分布
        states_tensor = torch.FloatTensor(np.array(states))
        masks_tensor = torch.FloatTensor(np.array(masks))
        policy_output = self.policy(states_tensor, masks_tensor)
        log_probs = torch.log(policy_output.gather(1, torch.LongTensor(actions).unsqueeze(-1)))
        ratios = torch.exp(log_probs - old_log_probs.detach())

        clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        
        # 更新价值网络
        values = self.value(torch.FloatTensor(states))
        value_loss = F.mse_loss(values.squeeze(), torch.FloatTensor(returns))

        # 计算熵的损失
        entropy_loss = -(policy_output * torch.log(policy_output + 1e-8)).sum(dim=-1).mean()

        # 更新策略
        self.optimizer_policy.zero_grad()
        (-self.entropy_coef * entropy_loss + policy_loss).backward(retain_graph=True)
        self.optimizer_policy.step()
        
        # 更新价值
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

    def train(self, env, num_episodes, max_steps, batch_size=32, buffer_size=10000):
        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        episode_transitions = []
        replay_buffer = []
        
        progress_bar = tqdm(range(num_episodes), desc="Training Progress")
        running_reward = 0
        running_length = 0
        best_reward = float('-inf')

        for episode in range(num_episodes):
            episode_start = time.time()
            state = env.reset()
            done = False
            episode_reward = 0
            t = 0
            while not done and t < max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, 117, 4, 9)
                # 调用 build_available_action_mask 生成动作掩码
                action_mask, _ = env.build_available_action_mask(state)
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)

                # # 获取模型输出
                # policy_output = self.policy(
                #     torch.FloatTensor(state).unsqueeze(0), 
                #     torch.FloatTensor(actionmask).unsqueeze(0)
                # )
                # action = torch.argmax(policy_output).item()  # 对模型输出调用 argmax
                
                # log_prob = torch.log(policy_output[0, action])  # 获取对应动作的 log 概率

                # next_state, reward, done, _ = env.step(action)
                # episode_reward += reward
                # replay_buffer.append((state, action, log_prob, reward, next_state, done))
                # if len(replay_buffer) > buffer_size:
                #     replay_buffer.pop(0)

                # 使用采样而非argmax进行探索
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
                    policy_output = self.policy(state_tensor, mask_tensor)
                    dist = Categorical(policy_output)
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action))
                    
                    # 计算状态值
                    value = self.value(state_tensor)
                
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # 存储整个转换（包括值估计和掩码）
                episode_transitions.append({
                    'state': state,
                    'action': action,
                    'log_prob': log_prob.item(),
                    'reward': reward,
                    'value': value.item(),
                    'mask': action_mask.copy(),
                    'done': done
                })
                state = next_state
                t += 1
            #visualization
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_length = 0.05 * t + (1 - 0.05) * running_length
            progress_bar.set_postfix({
                'Episode': f'{episode}/{num_episodes}',
                'Reward': f'{episode_reward:.2f}',
                'Avg Reward': f'{running_reward:.2f}',
                'Best': f'{best_reward:.2f}',
                'Length': f'{t}',
                'Time': f'{time.time() - episode_start:.2f}s'
            })
            # 计算整个回合的回报和优势
            returns = self.compute_returns(episode_transitions)
            advantages = self.compute_advantages(episode_transitions, returns)
            
            # 将回合数据添加到回放缓冲区
            for i, trans in enumerate(episode_transitions):
                if len(replay_buffer) >= buffer_size:
                    replay_buffer.pop(0)
                replay_buffer.append({
                    'state': trans['state'],
                    'action': trans['action'],
                    'old_log_prob': trans['log_prob'],
                    'return': returns[i],
                    'advantage': advantages[i],
                    'mask': trans['mask']
                })
            
            # 从回放缓冲区采样批次
            if len(replay_buffer) >= batch_size:
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]
                
                states = [item['state'] for item in batch]
                actions = [item['action'] for item in batch]
                old_log_probs = torch.tensor([item['old_log_prob'] for item in batch])
                returns = [item['return'] for item in batch]
                advantages = torch.tensor([item['advantage'] for item in batch])
                masks = [item['mask'] for item in batch]
                
                self.train_step(states, actions, old_log_probs, returns, advantages, masks)

            print("Episode: {}, Reward: {}".format(episode, episode_reward))


    def compute_returns(self, transitions):
        """计算折扣回报"""
        returns = []
        R = 0
        for trans in reversed(transitions):
            R = trans['reward'] + self.gamma * R * (1 - trans['done'])
            returns.insert(0, R)
        return returns

    def compute_advantages(self, transitions, returns):
        """计算优势函数"""
        values = np.array([trans['value'] for trans in transitions])
        advantages = returns - values
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    

# 定义并训练麻将机器人
from collections import namedtuple

class env_run():
    def __init__(self, action_model, memory_size=10000):  # 添加 memory_size 参数
        self.action_model = action_model

        # 定义 observation_space 和 action_space
        self.observation_space = namedtuple('Space', ['shape'])(shape=(117, 4, 9))  # 根据状态的维度调整
        self.action_space = namedtuple('Space', ['n'])(n=235)  # 动作空间大小为 233+ 1 PASS

        # 传递 memory_size 参数
        self.memory = MahjonggMemory(max_size=memory_size)
        self.action_model.eval()
        self.reset()

    def reset(self):
        self.hand = np.zeros((4, 9, 1), dtype=np.int)
        self.fulu = np.zeros((4, 9, 4), dtype=np.int)  # 修复拼写错误
        self.discard = np.zeros((4, 9, 112), dtype=np.int)  # 修复拼写错误
        self.fan_count = 0
        self.turnID = 0
        self.tile_count = [21, 21, 21, 21]
        self.myPlayerID = 0
        self.quan = 0
        self.prev_request = ''
        self.an_gang_card = ''
        self.discardnumber = np.zeros((4, 1), dtype=np.int)
        self.chownumber = np.zeros((4, 1), dtype=np.int)
        self.hand_fixed_data = []
        self.state = []

        # 返回初始状态
        # return np.concatenate([self.hand, self.fulu, self.discard], axis=2)
        state = np.concatenate([self.hand, self.fulu, self.discard], axis=2)
        state = np.transpose(state, (2, 0, 1))
        return state

    def step(self, request=None, response_target=None, fname=None):
        if fname:
            self.fname = fname


        inputJSON = json.loads(input())
        request = inputJSON['requests'][-1].split(' ')


        request = self.build_hand_history(request)
        if self.turnID <= 1:
            response = 'PASS'
        else:
            def make_decision(action_probs,actionmask):
                action_probs=action_probs.detach().numpy()
                action_probs=action_probs*actionmask
                action=np.argmax(action_probs[34:236])
                card = np.argmax(action_probs[0:33])


                return action,card


            actionmask, cardmask = self.build_available_action_mask(request)




            state=torch.cat([self.hand,self.fulu,self.discard],dim=2)
            action_probs = self.action_model(state)
            action_num,card_num = make_decision(action_probs,actionmask)


            response = self.build_output(action_num,card_num)


        self.prev_request = request
        self.turnID += 1


        return response

    def build_hand_history(self, request):
        # 第0轮，确定位置
        if self.turnID == 0:
            _, myPlayerID, quan = request
            self.myPlayerID = int(myPlayerID)
            self.other_players_id = [(self.myPlayerID - i) % 4 for i in range(4)]
            self.player_positions = {}
            for position, id in enumerate(self.other_players_id):
                self.player_positions[id] = position
            self.quan = int(quan)
            return request


        # 第一轮，发牌
        if self.turnID == 1:
            for i in range(5, 17):
                (cardname,cardid) = self.getcard(request[i])
                j=0
                while self.hand[cardname,cardid,j]!=0:
                    j=j+1
                self.hand[cardname,cardid,j]=1

            return request
        if int(request[0]) == 3:
            request[0] = str(requests[request[2]].value)
        elif int(request[0]) == 2:
            request.insert(1, self.myPlayerID)
        request = self.maintain_status(request)
        state1=torch.cat([self.hand,self.fulu,self.discard],dim=2)
        self.state
        return request

    def getcard(self, card):
        if isinstance(card, (int, np.integer)):
            card_type = card // 9  # 获取牌的类型
            card_number = card % 9  # 获取牌的编号
            print(f"Card: {card}, Card type: {card_type}, Card number: {card_number}")
            return (card_type, card_number)
        if (cards[card[0]].value==4):
            return (cards[card[0]].value, int(card[1]) +3)
        return (cards[card[0]].value  ,int(card[1]) - 1)


    def maintain_status(self, request):
        requestID = int(request[0])
        playerID = int(request[1])
        player_position = self.player_positions[playerID]

        if requestID == 2:
            (cardname, cardid) = self.getcard(request[-1])
            j = 0
            while self.hand[cardname, cardid, j] != 0:
                j = j + 1
            self.hand[cardname, cardid, j] = 1

        elif requests(requestID) == requests.PLAY:
            (cardname, cardid) = self.getcard(request[-1])

            # 自己
            if player_position == 0:
                #手牌更新
                j = 0
                while self.hand[cardname, cardid, j] != 0:
                    j = j + 1
                self.hand[cardname, cardid, j-1] = 0
            #打出的牌的序列
            self.discard[cardname, cardid, player_position*24+self.discardnumber[player_position]] = 1
            self.discardnumber[player_position]+=1
            #打出的牌的历史记录
            k = 96+player_position*4
            while self.discard[cardname, cardid, 96+player_position*4+k] != 0:
                k = k + 1
            self.discard[cardname, cardid, k] = 1
        elif requests(requestID) == requests.PENG:

            (cardname, cardid) = self.getcard(request[-1])
            if player_position == 0:
                last_player = int(self.prev_request[1])
                last_player_pos = self.player_positions[last_player]
                self.hand_fixed_data.append(('PENG',cardname*9+ cardid , last_player_pos))
                j = 0
                while self.hand[cardname, cardid, j] != 0:
                    j = j + 1
                self.hand[cardname, cardid, j-1] = 0
                self.hand[cardname, cardid, j - 2] = 0
            for q in range(3):
                self.fulu[cardname, cardid, 32+player_position*4+q] = 1


            # 打出的牌的序列
            self.discard[cardname, cardid, player_position * 24 + self.discardnumber[player_position]] = 1
            self.discardnumber[player_position] += 1
            # 打出的牌的历史记录
            k = 96 + player_position * 4
            while self.discard[cardname, cardid, 96 + player_position * 4 + k] != 0:
                k = k + 1
            self.discard[cardname, cardid, k] = 1
            self.discard[cardname, cardid, k+1] = 1

        elif requests(requestID) == requests.CHI:

            middle_card, play_card = request[3:5]
            (midcardname, midcardid) = self.getcard(middle_card)
            (playcardname, playcardid)= self.getcard(play_card)
            (precardname, precardid) = self.getcard(self.prev_request[-1])

            if player_position == 0:
                self.hand_fixed_data.append(('CHI', midcardid+midcardname*9, precardid - midcardid + 2))
                #更新hand


                if(precardid-midcardid==-1):
                    j = 0
                    while self.hand[midcardname, midcardid, j] != 0:
                        j = j + 1
                    self.hand[midcardname, midcardid, j - 1] = 0
                    k = 0
                    while self.hand[midcardname, midcardid, k] != 0:
                        k = k + 1
                    self.hand[midcardname, midcardid, k - 1] = 0


                if (precardid - midcardid == 0):
                    j = 0
                    while self.hand[midcardname, midcardid-1, j] != 0:
                        j = j + 1
                    self.hand[midcardname, midcardid-1, j - 1] = 0
                    k = 0
                    while self.hand[midcardname, midcardid+1, k] != 0:
                        k = k + 1
                    self.hand[midcardname, midcardid+1, k - 1] = 0

                if (precardid - midcardid == 1):
                    j = 0
                    while self.hand[midcardname, midcardid, j] != 0:
                        j = j + 1
                    self.hand[midcardname, midcardid, j - 1] = 0
                    k = 0
                    while self.hand[midcardname, midcardid+1, k] != 0:
                        k = k + 1

                    self.hand[midcardname, midcardid+1, k - 1] = 0
            #更新fulu
            for q in range(3):

                self.fulu[midcardname, midcardid-1+q, player_position * 4 + self.chownumber[player_position]] = 1
            self.fulu[precardname, precardid, 16+player_position * 4 + self.chownumber[player_position]] = 1
            self.chownumber[player_position] += 1
            #更新discard

            self.discard[playcardname, playcardid, player_position * 24 + self.discardnumber[player_position]] = 1
            self.discardnumber[player_position] += 1
            # 打出的牌的历史记录
            k = 96 + player_position * 4
            while self.discard[playcardname, playcardid, 96 + player_position * 4 + k] != 0:
                k = k + 1
            self.discard[playcardname, playcardid, k] = 1
            if (precardid - midcardid == -1):
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid, k] = 1
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid+1, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid+1, k] = 1

            if (precardid - midcardid == 0):
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid-1, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid-1, k] = 1
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid + 1, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid + 1, k] = 1
            if (precardid - midcardid == 1):
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid-2, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid-2, k] = 1
                k = 96 + player_position * 4
                while self.discard[midcardname, midcardid - 1, 96 + player_position * 4 + k] != 0:
                    k = k + 1
                self.discard[midcardname, midcardid -1, k] = 1


        elif requests(requestID) == requests.GANG:
            # 暗杠
            if requests(int(self.prev_request[0])) in [requests.drawCard, requests.DRAW]:
                request[2] = requests.ANGANG.name
                if player_position == 0:

                    (precardname, precardid) = self.getcard(self.prev_request[-1])
                    self.hand_fixed_data.append(('GANG',precardname*9+ precardid , 0))
                    #更新hand
                    j = 0
                    while self.hand[precardname, precardid, j] != 0 and j<4:
                        self.hand[precardname, precardid, j] = 0
                        j = j + 1
                    #更新fulu
                    for i in range(3):
                        self.fulu[precardname, precardid, i+64] = 1

                    #更新discard
                    self.discard[precardname, precardid, player_position * 24 + self.discardnumber[player_position]] = 1
                    self.discardnumber[player_position] += 1
                    for i in range(3):
                        self.discard[precardname, precardid , 96 + player_position * 4 + i]=1



            else:
                # 明杠
                (precardname, precardid) = self.getcard(self.prev_request[-1])
                request[2] = requests.MINGGANG.name
                if player_position == 0:

                    (precardname, precardid) = self.getcard(self.prev_request[-1])
                    last_player = int(self.prev_request[1])
                    self.hand_fixed_data.append(
                        ('GANG', precardname*9+precardid, self.player_positions[last_player]))
                    #更新hand
                    j = 0
                    while self.hand[precardname, precardid, j] != 0 and j<4:
                        self.hand[precardname, precardid, j] = 0
                        j = j + 1
                    #更新fulu
                    for i in range(3):
                        self.fulu[precardname, precardid, i + 48+player_position*4] = 1


                else:
                    (precardname, precardid) = self.getcard(self.prev_request[-1])
                    #更新fulu
                    for i in range(3):
                        self.fulu[precardname, precardid, i + 48 + player_position * 4] = 1

                # 更新discard
                self.discard[
                    precardname, precardid, player_position * 24 + self.discardnumber[player_position]] = 1
                self.discardnumber[player_position] += 1
                for i in range(2):
                    self.discard[precardname, precardid, 96 + player_position * 4 + i] = 1




        elif requests(requestID) == requests.BUGANG:
            (cardname, cardid) = self.getcard(request[-1])
            if player_position == 0:
                #更新hand
                for id, comb in enumerate(self.hand_fixed_data):
                    if comb[1] == request[-1]:
                        self.hand_fixed_data[id] = ('GANG', comb[1], comb[2])
                        break

                j = 0
                while self.hand[cardname, cardid, j] != 0:
                    j = j + 1
                self.hand[cardname, cardid, j - 1] = 0
            #更新fulu
            for q in range(3):
                self.fulu[cardname, cardid, 16+player_position * 4 +q] = 0
            for w in range(3):
                self.fulu[cardname, cardid, 48+player_position * 4 +w] = 1
            #更新discard
            for i in range(3):
                self.discard[cardname, cardid, 96 + player_position * 4 + i] = 1
        return request


    def build_available_action_mask(self, request):
        # print(f"Debug: Received request: {request}, type: {type(request)}")
        # # 检查 request 是否有效
        if request is None or not isinstance(request, (list, np.ndarray)) or len(request) == 0:
            raise ValueError("Invalid request: request must be a non-empty list or array.")

        # 如果是 NumPy 数组，将其转换为列表
        if isinstance(request, np.ndarray):
            request = request.tolist()

        # 如果 request 是嵌套列表，提取第一层的第一个元素
        while isinstance(request[0], (list, np.ndarray)):
            request = request[0]

        try:
            requestID = int(request[0])
        except (ValueError, TypeError, IndexError):
            raise ValueError(f"Invalid request[0]: {request[0]} cannot be converted to an integer.")

        cardmask = np.zeros((4, 9), dtype=int)
        actionmask = np.zeros(235, dtype=np.float32)

        playerID = int(request[1])
        myPlayerID = self.myPlayerID

        (last_card, lastcardid) = self.getcard(request[-1])

        # 摸牌回合
        if requestID == 2:

            for i in range(3):
                for j in range(8):
                    for k in range(3):
                        a=0
                        if self.hand[i,j,k]==1:
                            cardmask[i,j]=1
                            a+=1
                        if a==3:
                            actionmask[131+9*i+j]=1
                            actionmask[199+9*i+j]=1

            # 杠上开花
            if requests(int(self.prev_request[0])) in [requests.ANGANG, requests.BUGANG]:

                isHu = self.judgeHu(last_card, playerID, True)
            # 这里胡的最后一张牌其实不一定是last_card，因为可能是吃了上家胡，需要知道上家到底打的是哪张
            else:
                isHu = self.judgeHu(last_card, playerID, False)
            if isHu >= 8:
                actionmask[233] = 1
                self.fan_count = isHu

        else:
            #pass
            actionmask[234] = 1
            # 别人出牌
            if requests(requestID) in [requests.PENG, requests.CHI, requests.PLAY]:
                if playerID != myPlayerID:
                    #判断碰，杠
                    a=0
                    for i in range(3):
                        if self.hand[last_card, lastcardid, i] == 1:
                            a += 1
                    if a == 3:
                        actionmask[97 + 9 * last_card + lastcardid] = 1

                    if a == 2:
                        actionmask[165 + 9 * last_card + lastcardid] = 1
                    # 判断吃
                    if last_card < 3:
                        if lastcardid == 1:
                            if self.hand[last_card, lastcardid + 1, 0] == 1 and self.hand[last_card, lastcardid + 2, 0] == 1:
                                actionmask[34 + last_card * 21] = 1

                        if lastcardid == 9:
                            if self.hand[last_card, lastcardid - 1, 0] == 1 and self.hand[last_card, lastcardid - 2, 0] == 1:
                                actionmask[34 + last_card * 21] = 1

                        if lastcardid==2:
                            if self.hand[last_card, lastcardid-1,0]==1 and self.hand[last_card, lastcardid+1,0]==1:
                                actionmask[34+last_card*21+1]=1
                            if self.hand[last_card, lastcardid+1,0]==1 and self.hand[last_card, lastcardid+2,0]==1:
                                actionmask[34+last_card*21+2]=1
                        if lastcardid==8:
                            if self.hand[last_card, lastcardid-1,0]==1 and self.hand[last_card, lastcardid+1,0]==1:
                                actionmask[34+last_card*21+18]=1
                            if self.hand[last_card, lastcardid+1,0]==1 and self.hand[last_card, lastcardid+2,0]==1:
                                actionmask[34+last_card*21+19]=1

                        else :
                            if self.hand[last_card, lastcardid - 1, 0] == 1 and self.hand[
                                last_card, lastcardid - 2, 0] == 1:
                                actionmask[34 + last_card * 21 + 3+(lastcardid-3)*3] = 1
                            if self.hand[last_card, lastcardid -1, 0] == 1 and self.hand[
                                last_card, lastcardid + 1, 0] == 1:
                                actionmask[34 + last_card * 21 + 3+(lastcardid-3)*3+1] = 1
                            if self.hand[last_card, lastcardid + 1, 0] == 1 and self.hand[
                                last_card, lastcardid + 2, 0] == 1:
                                actionmask[34 + last_card * 21 + 3+(lastcardid-3)*3+2] = 1


                    # 是你必须现在决定要不要抢胡
                    isHu = self.judgeHu(last_card, lastcardid, playerID, False, dianPao=True)
                    if isHu >= 8:
                        actionmask[233] = 1
                        self.fan_count = isHu
            # 抢杠胡
            if requests(requestID) == requests.BUGANG and playerID != myPlayerID:
                isHu = self.judgeHu(last_card, lastcardid, playerID, True, dianPao=True)
                if isHu >= 8:
                    actionmask[233] = 1
                    self.fan_count = isHu
        return actionmask, cardmask



    def judgeHu(self, last_card, lastcardid, playerID, isGANG, dianPao=False):
        hand = []
        for i in range(3):
            for j in range(8):
                if self.hand[i,j,0]==1:
                    k=0
                    while self.hand[i,j,k]==1:
                        hand.append(i*9+k)
                        k+=1
        a=0
        isJUEZHANG = False
        for i in range(16):
            if self.discard[last_card, lastcardid,84+i]==1:
                a+=1
                if a==4:
                    isJUEZHANG = True
                    break


        if self.tile_count[(playerID + 1) % 4] == 0:
            isLAST = True
        else:
            isLAST = False

        try:
            ans = MahjongFanCalculator(tuple(self.hand_fixed_data), tuple(hand), last_card*9+lastcardid, 0, playerID==self.myPlayerID,
                                       isJUEZHANG, isGANG, isLAST, self.myPlayerID, self.quan)
        except Exception as err:
            if str(err) == 'ERROR_NOT_WIN':
                return 0
            else:
                if not self.botzone:
                    print(hand, last_card, self.hand_fixed_data)
                    print(self.fname)
                    print(err)
                    return 0
        else:
            fan_count = 0
            for fan in ans:
                fan_count += fan[0]
            return fan_count

    def build_output(self, action, card):
        if action>=0 and action<=32:
            return 'play {}'.format(card)
        if action>=33 and action<=96:
            action -= 33
            i=int(action/3)
            if i==1:
                if action==0:
                    return 'CHI {} {}'.format(self.getcardname(1), self.getcardnamecard)
                if action==1:
                    return 'CHI {} {}'.format(self.getcardname(1), self.getcardname(card))
                if action==2:
                    return 'CHI {} {}'.format(self.getcardname(2), self.getcardname(card))

                else:
                    action-=3
                    j=int((action+1)/3)

                    action-=3*j

                    if j==6:
                        if action==0:
                            return 'CHI {} {}'.format(self.getcardname(j + 1), self.getcardname(card))
                        if action==2:
                            return 'CHI {} {}'.format(self.getcardname(j + 1), self.getcardname(card))
                        if action==1:
                            return 'CHI {} {}'.format(self.getcardname(j) , self.getcardname(card))
                    else:
                        if action==0:
                            return 'CHI {} {}'.format(self.getcardname(j+2), self.getcardname(card))

                        if action==2:
                            return 'CHI {} {}'.format(self.getcardname(j + 1), self.getcardname(card))
                        if action==1:
                            return 'CHI {} {}'.format(self.getcardname(j) , self.getcardname(card))


            if i==2:
                action-=21
                if action == 0:
                    return 'CHI {} {}'.format(self.getcardname(10), self.getcardname(card))
                if action == 1:
                    return 'CHI {} {}'.format(self.getcardname(10), self.getcardname(card))
                if action == 2:
                    return 'CHI {} {}'.format(self.getcardname(11), self.getcardname(card))

                else:
                    action -= 3
                    j = int((action + 1) / 3)

                    action -= 3 * j

                    if j == 6:
                        if action == 0:
                            return 'CHI {} {}'.format(self.getcardname(j + 10), self.getcardname(card))
                        if action == 2:
                            return 'CHI {} {}'.format(self.getcardname(j + 10), self.getcardname(card))
                        if action == 1:
                            return 'CHI {} {}'.format(self.getcardname(j+9), self.getcardname(card))
                    else:
                        if action == 0:
                            return 'CHI {} {}'.format(self.getcardname(j + 11), self.getcardname(card))

                        if action == 2:
                            return 'CHI {} {}'.format(self.getcardname(j + 10), self.getcardname(card))
                        if action == 1:
                            return 'CHI {} {}'.format(self.getcardname(j+9), self.getcardname(card))


            if i==3:
                action-=42
                if action == 0:
                    return 'CHI {} {}'.format(self.getcardname(19), self.getcardname(card))
                if action == 1:
                    return 'CHI {} {}'.format(self.getcardname(19), self.getcardname(card))
                if action == 2:
                    return 'CHI {} {}'.format(self.getcardname(20), self.getcardname(card))

                else:
                    action -= 3
                    j = int((action + 1) / 3)

                    action -= 3 * j

                    if j == 6:
                        if action == 0:
                            return 'CHI {} {}'.format(self.getcardname(j + 19), self.getcardname(card))
                        if action == 2:
                            return 'CHI {} {}'.format(self.getcardname(j + 19), self.getcardname(card))
                        if action == 1:
                            return 'CHI {} {}'.format(self.getcardname(j+18), self.getcardname(card))
                    else:
                        if action == 0:
                            return 'CHI {} {}'.format(self.getcardname(j + 20), self.getcardname(card))

                        if action == 2:
                            return 'CHI {} {}'.format(self.getcardname(j + 19), self.getcardname(card))
                        if action == 1:
                            return 'CHI {} {}'.format(self.getcardname(j+18), self.getcardname(card))


        if action >= 97 and action <=130:
            action -= 97
            return 'PENG {} {}'.format(self.getcardname(action), self.getcardname(card))

        if action >=131 and action <=164:
            action -= 131
            return 'GANG {} '.format(self.getcardname(action))
        if action >=165 and action <=198:
            return 'GANG'
        if action >=199 and action <=232:
            action -=199
            return 'BUGANG {} '.format(self.getcardname(action))
        if action==233:
            return 'HU'
        if action==234:
            return 'PASS'

    def getcardname(self,card):
        a=int(card/4)
        return cards(a).name+str(card%4+1)










