# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Basic random agent for DeepMind Lab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import time
import deepmind_lab
import cv2
import sys, pygame

def run(length, width, height, fps, level, record, demo, demofiles, video):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }
  if record:
    config['record'] = record
  if demo:
    config['demo'] = demo
  if demofiles:
    config['demofiles'] = demofiles
  if video:
    config['video'] = video
  env = deepmind_lab.Lab(level, ['VEL.TRANS', 'VEL.ROT', 'RGB_INTERLEAVED'], config=config)

  env.reset()

  # Setup pygame
  pygame.init()
  screen = pygame.display.set_mode((width, height))
  clock = pygame.time.Clock()


  seq_idx = 0
  data_idx = 0

  seq_size = 5

  split_every = 2000
  file_idx = 0

  file_num = int(length / split_every)
  out_images = np.zeros((split_every, seq_size, height, width), dtype=np.uint8)
  out_labels = np.zeros((split_every, seq_size, 6), dtype=np.float)

  for i in range(0, length*seq_size):
    clock.tick(fps*2)
    action = np.array([0, 0, 0, 0,0,0,0], dtype=np.intc)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            exit()
    if pygame.key.get_pressed()[pygame.K_LEFT]:
        action += np.array([-20, 0, 0, 0,0,0,0], dtype=np.intc)
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
            action += np.array([20, 0, 0, 0,0,0,0], dtype=np.intc)
    if pygame.key.get_pressed()[pygame.K_UP]:
            action += np.array([0, 0, 0, 1,0,0,0], dtype=np.intc)
    if pygame.key.get_pressed()[pygame.K_DOWN]:
            action += np.array([0, 0, 0, -1,0,0,0], dtype=np.intc)
    if pygame.key.get_pressed()[pygame.K_SPACE]:
            action += np.array([0, 0, 0, 0,0,1,0], dtype=np.intc)
    print("File: {}/{}, {}/{}".format(file_idx+1, file_num, data_idx+1, split_every))

    obs = env.observations()
    out_images[data_idx, seq_idx] = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_RGB2GRAY)
    img = np.transpose(obs['RGB_INTERLEAVED'], (1,0,2))
    out_labels[data_idx, seq_idx, 0:3] = obs['VEL.TRANS']
    out_labels[data_idx, seq_idx, 3:] = obs['VEL.ROT']

    seq_idx += 1
    if seq_idx >= seq_size:
        seq_idx = 0
        data_idx += 1
        if data_idx >= split_every:
            data_idx = 0
            file_idx += 1
            np.save("/media/oli/LinuxData/datasets/dml/img_{}.npy".format(file_idx), \
                    out_images)
            np.save("/media/oli/LinuxData/datasets/dml/label_{}.npy".format(file_idx), \
                    out_labels)

    reward = env.step(action, num_steps=1)
    pygame.surfarray.blit_array(screen, img)
    pygame.display.flip()

  print("Done")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='tests/empty_room_test',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
