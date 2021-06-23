import os
import pygame
import math

import pymunk
import random
from collections import deque
import torch
import numpy as np
import torch
import tqdm

from shapely.geometry import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# run on terminal ' tensorboard --logdir=runs '
from torch.utils.tensorboard import SummaryWriter

print(f"PyTorch Version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"currently available device: {device}")
print("max gpu memory allocation ", torch.cuda.max_memory_allocated())

pygame.init()

FPS = 100

left = 45
right = 1655
top = 45
bottum = 1155
const_space = round(math.sqrt((right**2)+(bottum**2)), 4)
print("constant ================= ", const_space)
fish_number = 1
score_board = [x for x in range(0, fish_number*10000, int(30**2))]


###########q_learning#############
Max_Memory = 20000
Batch_Size = 25
Long_Batch_Size = 10000
Mid_memory = 12
mid_Batch_Size = 11

LR = 0.0001
numbre_of_episode = 50000
iterr = 50000

version = 100
#################################

screen = pygame.display.set_mode((right+900, bottum+150), pygame.RESIZABLE)
font = pygame.font.SysFont("Times New Roman,Arial", 34)
font1 = pygame.font.SysFont("Times New Roman,Arial", 21)
clock = pygame.time.Clock()


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0).to(device)


class Fish(object):
    def __init__(self, x, y):

        self.body = pymunk.Body()
        self.body.position = int(x), int(y)
        self.body.velocity = 0, 0
        self.shape = pymunk.Circle(self.body, 22)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.collision_type = 2
        space.add(self.body, self.shape)
        self.imgIcon = 'icons/img1.png'
        self.clos_from_fish = 0
        self.clos_from_thebox = 0

    def draw(self):

        pygame.draw.circle(screen, (30, 50, 50),
                           self.body.position, 22)  # (50, 50, 50)

    def iconCreat(self):

        return pygame.image.load(self.imgIcon)

    def imgShow(self, display_screen, imag_fish):

        x, y = self.body.position
        display_screen.blit(imag_fish, (x-22, y-22))

    def if_inthebox(self):

        x, y = self.body.position
        if (int(x) in range(300, 500)) and (int(y) in range(300, 500)):

            return True
        else:

            return False

    def if_game_done(self):

        x, y = self.body.position
        if (int(x) in range(360, 440)) and (int(y) in range(360, 440)):

            return True
        else:

            return False

    def update_data(self, v, m):

        self.clos_from_fish = v
        self.clos_from_thebox = m

    def eco_data(self):

        x, y = (self.body.position)
        a, b = (self.body.velocity)
        c = self.clos_from_fish
        d = self.clos_from_thebox

        return [round(x/1705, 6), round(y/1205, 6), round(abs(a)/1705, 6), round(abs(b)/1205, 6), round(c, 6), round(d, 6)]


class AgentAI(object):

    def __init__(self, x, y, num_episode, iterr, number_of_fishs):
        self.bodyAI = pymunk.Body()
        self.bodyAI.position = int(x), int(y)
        self.bodyAI.velocity = 0, 0
        self.shape = pymunk.Circle(self.bodyAI, 22)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.collision_type = 2
        self.imgIcon = 'icons/img3.png'
        space.add(self.bodyAI, self.shape)
        #################################################
        self.iterr = iterr
        self.epsilon = 0
        self.gamma = 0.8
        
        self.memory = deque(maxlen=Max_Memory)
        self.mid_memory = deque(maxlen=Mid_memory)
        self.model = DQN((number_of_fishs*6)+6)
        self.num_episode = num_episode
        self.reward = 0.0
        self.trainer = Qtrainer(
            self.model, lr=LR, gamma=self.gamma)
        print(self.gamma)

    def IAdraw(self):

        # (150, 150, 150)
        pygame.draw.circle(screen, (30, 50, 50), self.bodyAI.position, 22)

    def remove_body(self):

        space.remove(self.bodyAI, self.shape)

    def rest_position(self, pos1, pos2, iterr):

        self.bodyAI = pymunk.Body()
        self.iterr = iterr
        self.bodyAI.position = pos1, pos2
        self.bodyAI.velocity = 0, 0
        self.shape = pymunk.Circle(self.bodyAI, 22)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.collision_type = 2
        self.imgIcon = 'icons/img3.png'
        space.add(self.bodyAI, self.shape)
        self.reward = 0.0

    def IAiconCreat(self):

        return pygame.image.load(self.imgIcon)

    def IAimgShow(self, display_screen, imag_fish):

        x, y = self.bodyAI.position
        display_screen.blit(imag_fish, (x-20, y-20))

    def IAeco_data(self, v, m):

        x, y = (self.bodyAI.position)
        a, b = (self.bodyAI.velocity)

        return [round(x/1705, 6), round(y/1205, 6), round(abs(a)/1705, 6), round(abs(b)/1205, 6), round(v, 6), round(m, 6)]

    def reward_update(self, ifinthebox, scoreBoard):

        return scoreBoard[ifinthebox]

    def check_position(self, f_close, b_close, local_score, iterr):
        x, y = self.bodyAI.position

        if ((int(x) not in range(85, 1610)) or (int(y) not in range(85, 1110))) or ((int(x) in range(860, 940)) and (int(y) in range(50, 715))):
            if self.reward > -2:
                self.reward -= 0.1
                return False, self.reward
            else:
                return True, self.reward

        score = (15*(((1-f_close)*2.5)+((1-b_close)*3.5)))

        self.reward = round((score/1.5) + (local_score/2.2), 6)

        return False, self.reward

    def move_Agent(self, prediction):

        tnp = np.array(prediction, dtype=np.int32)

        speed = FPS

        move_table = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1]
                               ])

        if (tnp == move_table[0]).all():  # up
            self.bodyAI.velocity = (0, -speed)

        elif (tnp == move_table[1]).all():  # down
            self.bodyAI.velocity = (0, speed)

        elif (tnp == move_table[2]).all():  # left
            self.bodyAI.velocity = (-speed, 0)

        elif (tnp == move_table[3]).all():  # right
            self.bodyAI.velocity = (speed, 0)

        elif (tnp == move_table[4]).all():  # down right
            self.bodyAI.velocity = (speed, speed)
        elif (tnp == move_table[5]).all():  # down left
            self.bodyAI.velocity = (-speed, speed)
        elif (tnp == move_table[6]).all():  # up right
            self.bodyAI.velocity = (speed, -speed)
        elif (tnp == move_table[7]).all():  # up left
            self.bodyAI.velocity = (-speed, -speed)

    def get_action(self, state, it, ep):
        self.epsilon = 220000+(ep*1000)

        final_move = [0., 0., 0., 0., 0., 0., 0., 0.]

        state0 = state.clone().detach()
        prediction = self.model(state0).clone().detach().cpu()

        move = torch.argmax(prediction).item()
        tnp = np.array(prediction)
        vv = np.count_nonzero(tnp == np.max(tnp))
        if sum(prediction) != 0 and vv == 1 and torch.max(prediction) > 0:
            final_move[move] = 1
            pred = torch.tensor(final_move, dtype=torch.float32).cpu()
            self.move_Agent(pred)
        return prediction.clone().detach().to(device)

    def remember(self, local_ts, action, xreward, xrecord, new_local_ts):
        self.memory.append((local_ts, action, xreward, xrecord, new_local_ts))

    def mid_remember(self, local_ts, action, xreward, xrecord, new_local_ts):
        self.mid_memory.append(
            (local_ts, action, xreward, xrecord, new_local_ts))

   
    def train_long_memory(self):

        if len(self.memory) > Batch_Size:
            mini_sample = random.sample(self.memory, Batch_Size)
        else:
            mini_sample = self.memory
        local_tss, actions, rewards, records, new_local_tss = zip(*mini_sample)

        self.trainer.train_step(
            local_tss, actions, rewards, records, new_local_tss)

    def train_long_memory_2(self):

        if len(self.memory) > Long_Batch_Size:
            mini_sample = random.sample(self.memory, Long_Batch_Size)
        else:
            mini_sample = self.memory
        local_tss, actions, rewards, records, new_local_tss = zip(*mini_sample)

        self.trainer.train_step(
            local_tss, actions, rewards, records, new_local_tss)

    def train_mid_memory(self):
        if len(self.mid_memory) > mid_Batch_Size:
            mini_sample1 = random.sample(self.mid_memory, mid_Batch_Size)
        else:
            mini_sample1 = self.mid_memory
        local_tss, actions, rewards, records, new_local_tss = zip(
            *mini_sample1)

        self.trainer.train_step(
            local_tss, actions, rewards, records, new_local_tss)
        

    

class Wall(object):
    def __init__(self, p1, p2, collision_number=None, thikness=None):
        self.thikness = thikness
        self.bodyW = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.bodyW, p1, p2, self.thikness)
        self.shape.elasticity = 0.8
        space.add(self.shape, self.bodyW)
        if collision_number:
            self.shape.collision_type = collision_number

    def draw(self):
        pygame.draw.line(screen, (180, 150, 150),
                         self.shape.a, self.shape.b, self.thikness)


def collect_data(local_fishs, local_fish, v, m):

    lit = []

    for ff in local_fishs:
        lit.append(ff.eco_data())

    lit.append(local_fish.IAeco_data(v, m))

    lit_np = np.array(lit)

    lit_np = lit_np.flatten()

    lit_pytorch = torch.tensor(lit_np, dtype=torch.float32, device=device)

    return lit_pytorch


def show_output(action):

    p = torch.argmax(action).item()
    pygame.draw.circle(screen, (200, 50, 50), (1800, 800+(p*50)), 15)


def game(locale_numbre_of_episode, iterration):
    global space, LR

    show_line1 = False
    show_line2 = False
    show_data = False
    game_complited = 0

    reward = 0
    old_reward1 = 0
    space = pymunk.Space()
    fish = AgentAI(400, 400, locale_numbre_of_episode,
                   iterration, fish_number)
    if os.path.isfile('./model/model_{}_game_done.pth'.format(version)):

        fish.model.load()
    for epi in tqdm.tqdm(range(locale_numbre_of_episode)):

        record = 0

        fishs = [Fish(random.randint(1100, right-200),
                      random.randint(100, bottum-100)) for _ in range(fish_number)]

        fish_image = fishs[0].iconCreat()

        IAfish_image = fish.IAiconCreat()
        IAfish_image = pygame.transform.scale(
            IAfish_image, (44, 44))

        fish_image = pygame.transform.scale(
            fish_image, (44, 44))

        wall_left = Wall([left, top], [left, bottum], 2, 30)
        wall_right = Wall([right, top], [right, bottum], 2, 30)
        wall_top = Wall([left, top], [right, top], 2, 30)
        wall_bottum = Wall([left, bottum], [right, bottum], 2, 30)
        wall_1 = Wall([900, 50], [900, 700], 2, 30)
        line_wall = LineString(([900, 50], [900, 700]))
        fish_inthebox = None

        fish.n_game = iterration

        iterr = iterration

        game_done = False
        fish_inthebox = 0
        fish_game_done = 0
        close_from_fish = []
        close_form_thebox = []

        clos_to_fiches = 0
        clos_to_box = 0

        for ff in fishs:
            ff.draw()
            ff.imgShow(screen, fish_image)

            if ff.if_inthebox() == True:
                fish_inthebox += 1
            if ff.if_game_done() == True:
                fish_game_done += 1

            ff.body.velocity = (
                (ff.body.velocity[0])/1.004), ((ff.body.velocity[1])/1.004)
            line1 = LineString([ff.body.position, fish.bodyAI.position])
            line2 = LineString([ff.body.position, (400, 400)])
            if not line1.intersects(line_wall):
                cf = round((math.sqrt(((ff.body.position[0]-fish.bodyAI.position[0])**2)+(
                    (ff.body.position[1]-fish.bodyAI.position[1])**2)))/const_space, 6)
            else:
                cf = 0

            if not line2.intersects(line_wall):
                cb = round(((math.sqrt(((ff.body.position[0]-400)**2)+(
                    (ff.body.position[1]-400)**2))))/const_space, 6)
            else:
                cb = 0

            ff.update_data(cf, cb)
            close_from_fish.append(cf)
            close_form_thebox.append(cb)

        clos_to_fiches = round(
            (sum(close_from_fish)/(fish_number)), 6)
        clos_to_box = round(((sum(close_form_thebox)/fish_number)), 6)
        old_state = collect_data(fishs, fish, clos_to_fiches, clos_to_box)
        while iterr > 1 and not game_done:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    return collect_data(fishs, fish, clos_to_fiches, clos_to_box)
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_KP5:

                        show_line1 = False

                    if event.key == pygame.K_KP4:
                        show_line1 = True

                    if event.key == pygame.K_KP8:
                        show_line2 = False

                    if event.key == pygame.K_KP7:
                        show_line2 = True
                    if event.key == pygame.K_KP1:
                        show_data = True
                    if event.key == pygame.K_KP2:
                        show_data = False
                    if event.key == pygame.K_KP0:
                        iterr = 3

            fish.num_episode = iterration
            pygame.display.update()

            clock.tick(FPS)
            space.step(1.5/FPS)
            screen.fill((30, 50, 50))

            pygame.draw.rect(screen, (80, 80, 80), pygame.Rect(
                300, 300, 200, 200), width=0)
            pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(
                360, 360, 80, 80), width=0)
            if show_data:
                pygame.draw.circle(screen, (130, 180, 150),
                                   fish.bodyAI.position, (const_space*12)/100, 2)
            wall_left.draw()
            wall_right.draw()
            wall_top.draw()
            wall_bottum.draw()
            wall_1.draw()
            fish.IAdraw()
            fish.IAimgShow(screen, IAfish_image)
            ##################################################old state########################################

            old_state1 = old_state.clone().detach().to(
                device)                   # old state

            ############################################################end colaction #######################################################

            action_now = fish.get_action(old_state, iterr, epi)  # Action
            action_now1 = action_now.clone().detach().to(device)

            ###################################################### new state################################################
            fish_inthebox = 0
            fish_game_done = 0
            clos_to_fiches = 0
            clos_to_box = 0

            close_from_fish = []
            close_form_thebox = []
            for ff in fishs:
                ff.draw()
                ff.imgShow(screen, fish_image)

                if ff.if_inthebox() == True:
                    fish_inthebox += 1
                if ff.if_game_done() == True:
                    fish_game_done += 1

                ff.body.velocity = (
                    (ff.body.velocity[0])/1.004), ((ff.body.velocity[1])/1.004)
                line1 = LineString([ff.body.position, fish.bodyAI.position])
                line2 = LineString([ff.body.position, (400, 400)])
                if not line1.intersects(line_wall):
                    cf = round((math.sqrt(((ff.body.position[0]-fish.bodyAI.position[0])**2)+(
                        (ff.body.position[1]-fish.bodyAI.position[1])**2)))/const_space, 6)
                else:
                    cf = 0

                if not line2.intersects(line_wall):
                    cb = round(((math.sqrt(((ff.body.position[0]-400)**2)+(
                        (ff.body.position[1]-400)**2))))/const_space, 6)
                else:
                    cb = 0
                ff.update_data(cf, cb)
                close_from_fish.append(cf)
                close_form_thebox.append(cb)
                if show_line1 and (not line1.intersects(line_wall)):
                    pygame.draw.aaline(screen, (180, 100, 50),
                                       ff.body.position, fish.bodyAI.position)
                if show_line2 and (not line2.intersects(line_wall)):

                    pygame.draw.aaline(screen, (180, 100, 50),
                                       (400, 400), ff.body.position)

            clos_to_fiches = round(
                (sum(close_from_fish)/(fish_number)), 6)
            clos_to_box = round(((sum(close_form_thebox)/fish_number)), 6)

            ########################################### end collaciton #############################################

            game_done, reward = fish.check_position(
                clos_to_fiches, clos_to_box, fish.reward_update(fish_inthebox, score_board), iterr)
            if fish_game_done == fish_number:

                game_done = True
                game_complited += 1
                
            reward1 = torch.tensor(
                [round((reward), 6)], dtype=torch.float32, device=device)

            new_state = collect_data(fishs, fish, clos_to_fiches, clos_to_box)

            new_state1 = new_state.clone().detach().to(device)  # new state

           
            if game_done:
                game_done1 = torch.tensor(
                    [1.0], dtype=torch.float32, device=device)
            else:
                game_done1 = torch.tensor(
                    [0.0], dtype=torch.float32, device=device)

            ###################################################################################################
            delta_reward = (reward1-old_reward1)

            if (100*clos_to_fiches <= 12 and clos_to_fiches > 0 and delta_reward > 0):
                delta_reward = (delta_reward*1.15)
            if (100*clos_to_fiches > 12 and clos_to_fiches > 0 and delta_reward > 0):
                delta_reward = (delta_reward)/0.85
            
            
            if delta_reward > record and (not line1.intersects(line_wall)) and iterr+2 < iterration:

                record = delta_reward

                fish.record = record
                fish.model.save()
            record1 = torch.tensor(
                [record], dtype=torch.float32, device=device)

            fish.remember(old_state1, action_now1, delta_reward.to(device), game_done1,
                          new_state1)            # remember
            fish.mid_remember(old_state1, action_now1, delta_reward.to(device), game_done1,
                              new_state1)
            
            ##################################################################################################

            if iterr % 30 == 0 and (delta_reward == 0):

                fish.train_long_memory()

            elif iterr % 11 == 0:
                fish.train_mid_memory()
                if iterr % 350 == 0:
                    fish.train_long_memory()

            ########################
            old_state = new_state

            old_reward1 = reward1
            #########################

            txt = np.array(action_now1.clone().detach().cpu(),
                           dtype=np.float16)
            f = []
            for i in txt:
                f.append(round(i, 4))
            txt = np.array(old_state.clone().detach().cpu(),
                           dtype=np.float16)

            s = []
            for i in txt:
                s.append(round(i, 4))
            xt = np.array(delta_reward.clone().detach().cpu(),
                          dtype=np.float32)
            ########################################################################################################
            color1 = (200, 200, 100)
            color2 = (210, 100, 10)
            out_put = ["--N", "--S", "--W", "--E",
                       "--SE", "--SW", "--NE", "--NW"]
            for i in range(len(out_put)):
                text_output = font1.render(
                    out_put[i], True, (200, 200, 100))
                screen.blit(text_output, (1815, 788+(i*50)))
                pygame.draw.circle(screen, (150, 150, 150),
                                   (1800, 800+(i*50)), 17)
            pygame.display.set_caption(
                "By Lotfi AlgerIA  *** version:{0}   ,score: {1} ,reward: {2}  ***".format(version, record, reward))
            text1 = font.render("Machine Learning ,Q_Learning Algorithm Version: 0.0{0} .".format(
                version), True, color1)
            screen.blit(text1, (1690, 100))

            text_info = font.render(
                "  Using Python ,Pytorch and CUDA.", True, color1)
            screen.blit(text_info, (1690, 150))
            text2 = font.render("Epoch number : {0}".format(
                epi+1), True, color2)
            screen.blit(text2, (1750, 250))

            text3 = font.render("Number of Epochs successfully complited : {0}".format(
                game_complited), True, color2)
            screen.blit(text3, (1750, 300))

            text3 = font.render("Iteration : {0}".format(
                iterr), True, color2)
            screen.blit(text3, (1750, 350))

            text4 = font.render(
                "Input data for the neural network:", True, color1)
            screen.blit(text4, (1750, 500))
            if iterr % 10 == 0:
                text5 = font1.render("{0}".format(s), True, color2)
                text7 = font1.render("{0}".format(f), True, color2)
            show_output(action_now.clone().detach().cpu())
            text6 = font.render(
                "Output of the neural network:", True, color1)
            screen.blit(text6, (1750, 600))

            screen.blit(text5, (1690, 550))
            screen.blit(text7, (1690, 650))

            if show_data:
                text8 = font1.render("T {0}".format(
                    round(100*clos_to_box, 2)), True, (130, 180, 150))
                for ff in fishs:
                    screen.blit(text8, ff.body.position+(-17, -42))
                text9 = font1.render("A {0}".format(
                    round(100*clos_to_fiches, 2)), True, (130, 180, 150))
                screen.blit(text9, fish.bodyAI.position+(-10, -42))
                text10 = font1.render("d: {0}, r:{1}".format(
                    round(100*xt[0], 2), round(reward, 2)), True, (130, 180, 150))
                screen.blit(text10, fish.bodyAI.position+(-50, 36))

            ########################################training step########################

            iterr -= 1
        fish.remove_body()
        space = pymunk.Space()
        fish.rest_position(400, 400, iterration)

        if os.path.isfile('./model/model_{}_game_done.pth'.format(version)):
            fish.model.load()

        fish.train_long_memory_2()
        if reward > record and (not line1.intersects(line_wall)):
            record = reward
            print(record)
            fish.record = record
            record1 = torch.tensor(
                [record], dtype=torch.float32, device=device)
            fish.model.save()
        fish.reward = 0
        reward = fish.reward


class DQN(nn.Module):
    def __init__(self, matrix_len):
        super().__init__()
        self.fc1 = nn.Linear(in_features=matrix_len,
                             out_features=225).to(device)
        self.fc2 = nn.Linear(in_features=225, out_features=225).to(device)
        ######
        self.fc3 = nn.Linear(in_features=225, out_features=280).to(device)

        self.fc4 = nn.Linear(in_features=280, out_features=280).to(device)
        ######
        self.fc5 = nn.Linear(in_features=280, out_features=144).to(device)
        self.fc6 = nn.Linear(in_features=144, out_features=36).to(device)

        self.out_fc = nn.Linear(in_features=36, out_features=8).to(device)

        ##############################################################

        self.pr1 = nn.LeakyReLU(negative_slope=0.025).to(device)
        self.pr2 = nn.LeakyReLU(negative_slope=0.025).to(device)
        self.pr3 = nn.GELU().to(device)
        self.pr4 = nn.GELU().to(device)
        self.pr5 = nn.GELU().to(device)  # nn.LeakyReLU(negative_slope=0.025).to(device)
        self.pr6 = nn.GELU().to(device)
        self.pr7 = nn.Softmax(dim=-1).to(device)
        ##############################################################

    def forward(self, t):

        t = 10*self.pr1(self.fc1(t)).to(device)
        t = self.pr2(self.fc2(t)).to(device)
        t = self.pr3(self.fc3(t)).to(device)
        t = 10*torch.sigmoid(self.fc4(t)).to(device)
        t = torch.sigmoid(self.fc5(t)).to(device)
        t = torch.sigmoid(self.fc6(t)).to(device)
        t = self.pr7(self.out_fc(t)).to(device)

        return t

    def save(self, file_name='model_{}_game_done.pth'.format(version)):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='./model/model_{}_game_done.pth'.format(version)):
        if os.path.exists(file_name):
            self.state_dict(torch.load(file_name))
            self.eval()


class Qtrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr, eps=0.00001, weight_decay=0.0001)
        self.criterion = nn.SmoothL1Loss(beta=0.2).to(device)  # 0.3 0.1

        self.writer = SummaryWriter(f'runs')
        self.step = 0

    def train_step(self, State, Action, reward, if_done, New_state):

        Action = tuple_of_tensors_to_tensor(Action)

        State = tuple_of_tensors_to_tensor(State)

        New_state = tuple_of_tensors_to_tensor(New_state)

        if isinstance(reward, tuple):

            Reward = tuple_of_tensors_to_tensor(reward)
            If_done = tuple_of_tensors_to_tensor(if_done)
            Reward = Reward.view(-1, 1)
            If_done = If_done.view(-1, 1)

            dqn_v = Reward.numel()

        else:
            If_done = if_done.clone().detach()
            Reward = reward.clone().detach()
            dqn_v = Reward.numel()

        if Reward.numel() == 1:

            Reward = torch.unsqueeze(Reward, 0)
            If_done = torch.unsqueeze(If_done, 0)

            New_state = torch.unsqueeze(New_state, 0)
            State = torch.unsqueeze(State, 0)
            Action = torch.unsqueeze(Action, 0)

        pred = self.model(State).to(device)

        target = pred.clone().to(device)

        for i in range(dqn_v):

            Q_new = Reward[i].to(device)

            if If_done[i] == 0:

                Q_new = (Reward[i]+(self.gamma *
                                    torch.max(self.model(New_state[i])))).to(device)  # -(1-self.gamma)*torch.max(self.model(State[i]))

            target[i, torch.argmax(Action[i]).item()] = Q_new.to(device)

        self.optimizer.zero_grad()

        loss = self.criterion(pred, target).to(device)

        loss.backward()

        self.optimizer.step()
        self.writer.add_scalar('Training Loss{0}'.format(
            version), loss, global_step=self.step)

        self.step += 100
        self.writer.flush()


if __name__ == '__main__':

    p = game(numbre_of_episode, iterr)

    pygame.quit()
