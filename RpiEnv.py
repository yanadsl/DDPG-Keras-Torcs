import gym
from gym import spaces
import numpy as np
# from os import path
import numpy as np
import copy
import collections as col
import os
import datetime
import time
import pigpio


class Env:
    left_servo_pin = 5
    right_servo_pin = 6
    sensor_signal_pin = 4
    sensor_message_size = 8
    time_record = int(time.time() * 1000)
    time_limit = 50
    def __init__(self,):
        self.pi = pigpio.pi()
        self.pi.set_mode(self.sensor_signal_pin, pigpio.OUTPUT)
        self.h1 = self.pi.serial_open("/dev/ttyAMA0")
        self.initial_run = True


    def step(self, action):
        if action == 'left':
            self.pi.set_servo_pulsewidth(self.left_servo_pin, 1500)
            self.pi.set_servo_pulsewidth(self.right_servo_pin, 1500)
        elif action == 'left':
            self.pi.set_servo_pulsewidth(self.left_servo_pin, 1500)
            self.pi.set_servo_pulsewidth(self.right_servo_pin, 1500)
        else:
            self.pi.set_servo_pulsewidth(self.left_servo_pin, 1500)
            self.pi.set_servo_pulsewidth(self.right_servo_pin, 1500)

    def get_respond(self):
        while (int(time.time() * 1000) - self.time_record) >= self.time_limit:
            time.sleep(0.002)
        self.time_record = int(time.time() * 1000)
        distance = []
        self.pi.serial_read(self.h1)    #clear any redauntancy data
        self.pi.write(self.sensor_signal_pin, pigpio.HIGH)
        while self.pi.serial_data_available(self.h1) < self.sensor_message_size:
            time.sleep(0.0007)
        (b, d) = self.pi.serial_read(self.h1, self.sensor_message_size)
        self.pi.write(self.sensor_signal_pin, pigpio.LOW)
        for a in d:
            distance.append(int.from_bytes(a, byteorder='big', signed=False))

        distance = self.normalize(distance)

        return distance

    def get_reward(self,state):
        if '0' in state:
            reward = 0
            dead = True
        else:
            reward = 10
            dead = False
        return reward, dead

    def normalize(self,things):
        dis = ''
        for distance in things:
            if distance >= 36:
                dis += '8'
            elif distance >= 24:
                dis += '7'
            elif distance >= 20:
                dis += '6'
            elif distance >= 16:
                dis += '5'
            elif distance >= 12:
                dis += '4'
            elif distance >= 10:
                dis += '3'
            elif distance >= 8:
                dis += '2'
            elif distance >= 6:
                dis += '1'
            else:
                dis += '0'
        return dis



    def end(self):
        self.pi.set_servo_pulsewidth(self.left_servo_pin, 0)
        self.pi.set_servo_pulsewidth(self.right_servo_pin, 0)