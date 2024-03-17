# -*- coding: utf-8 -*-
# @Time    : 2024/3/15 12:33
# @Author  : Yifan Zhang
# @File    : StanleyController.py

import glob
import os
import sys
import cv2
import math
import time

sys.path.append("D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla")
sys.path.append("D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\agents")


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("D:\\CARLA_0.9.15\\WindowsNoEditor\\carla_RL\\carla_control\\utils")
sys.path.append("..")

import carla
import numpy as np
import pygame
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import PIDLongitudinalController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector
from agents.tools.misc import get_speed
from utils.InitHelper import InitHelper
from utils.CameraHelper import CameraHelper

SHOW_CAM = True
distance = 2.0
T = 100

L = 2.875
Kdd = 4.0
alpha_prev = 0
delta_prev = 0

init_helper = InitHelper()

client = init_helper.connect2server()
world = init_helper.world
init_helper.set_world_settings()
spectator = init_helper.set_spectator_transform()
spawn_pts = init_helper.draw_spawn_points()
route = init_helper.global_path_planning(spawn_pts[88].location, spawn_pts[27].location)

wps = [waypoint[0] for waypoint in route]
init_helper.draw_waypoints(wps)

next = wps[0]
blueprint_library = world.get_blueprint_library()

# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
ego = world.spawn_actor(ego_bp, spawn_pts[88])
camera_helper = CameraHelper(world, ego)
camera_location = carla.Location(x=-5, z=3)
camera_rotation = carla.Rotation(pitch=-20)
camera = camera_helper.spawn_camera(camera_location, camera_rotation, camera_helper.show_image)

control = carla.VehicleControl()

waypoint_list = []

for wp in wps:
    waypoint_list.insert(wps.index(wp), (wp.transform.location.x, wp.transform.location.y))

pid = PIDLongitudinalController(ego, K_P=1, K_I=0.75, K_D=0.0, dt=0.01)


# Change the steer output with the lateral controller.
steer_output = 0

# Use stanley controller for lateral control
# 0. spectify stanley params
k_e = 0.3
k_v = 10

target_speed = 30

# Generate waypoints
i = 0

try:
    while True:
        ego_transform = ego.get_transform()
        ego_loc = ego.get_location()
        spectator.set_transform(
            carla.Transform(ego_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))

        world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=T)
        world.debug.draw_point(next.transform.location, color=carla.Color(r=255), life_time=T)
        ego_dist = distance_vehicle(next, ego_transform)

        yaw = np.radians(ego_transform.rotation.yaw)
        # 1. calculate heading error
        yaw_path = np.arctan2(waypoint_list[-1][1] - waypoint_list[0][1], waypoint_list[-1][0] - waypoint_list[0][0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi


        # 2. calculate crosstrack error
        current_xy = np.array([ego_loc.x, ego_loc.y])
        crosstrack_error = np.min(np.sum((current_xy - np.array(waypoint_list)[:, :2]) ** 2, axis=1))

        yaw_cross_track = np.arctan2(ego_loc.y - waypoint_list[0][1], ego_loc.x - waypoint_list[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        ego_vel = ego.get_velocity()

        vf = np.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)
        vf = np.fmax(np.fmin(vf, 2.5), 0.1)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + vf))

        print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

        # 3. control low
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < - np.pi:
            steer_expect += 2 * np.pi
        steer_expect = min(1.22, steer_expect)
        steer_expect = max(-1.22, steer_expect)

        # 4. update
        steer_output = steer_expect

        throttle = pid.run_step(target_speed)
        control = carla.VehicleControl()
        throttle = pid.run_step(target_speed)
        control = carla.VehicleControl()
        if throttle >= 0.0:
            control.throttle = min(throttle, 0.5)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(throttle), 0.3)

        control.steer = steer_output
        control.throttle = throttle

        ego.apply_control(control)


        if i == (len(wps) - 1):
            control.brake = 1
            ego.apply_control(control)
            print('this trip finish')
            time.sleep(10)
            break

        if ego_dist < 2:
            i = i + 1
            next = wps[i]
            ego.apply_control(control)

        # print(i)
        world.wait_for_tick()

finally:
    ego.destroy()
    camera.stop()
    pygame.quit()