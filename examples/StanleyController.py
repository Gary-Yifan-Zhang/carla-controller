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


def calculate_trajectory_yaw(waypoint_list, min_idx):
    if min_idx + 5 < len(waypoint_list):
        yaw = np.arctan2(waypoint_list[min_idx + 3][1] - waypoint_list[min_idx - 1][1],
                         waypoint_list[min_idx + 3][0] - waypoint_list[min_idx - 1][0])
    else:
        yaw = np.arctan2(waypoint_list[-1][1] - waypoint_list[min_idx - 1][1],
                         waypoint_list[-1][0] - waypoint_list[min_idx - 1][0])

    return yaw


def get_target_wp_index(veh_location, waypoint_list):
    dxl, dyl = [], []
    for i in range(len(waypoint_list)):
        dx = abs(veh_location.x - waypoint_list[i][0])
        dxl.append(dx)
        dy = abs(veh_location.y - waypoint_list[i][1])
        dyl.append(dy)

    dist = np.hypot(dxl, dyl)
    idx = np.argmin(dist) + 4

    # take closest waypoint, else last wp
    if idx < len(waypoint_list):
        tx = waypoint_list[idx][0]
        ty = waypoint_list[idx][1]
    else:
        tx = waypoint_list[-1][0]
        ty = waypoint_list[-1][1]

    return idx, tx, ty, dist


def get_global_yaw(start_point, end_point):
    """
    计算全局航向角（yaw），即起始点到结束点的方位角（角度）
    Args:
        start_point: 起始点的坐标，形式为 [x, y]
        end_point: 结束点的坐标，形式为 [x, y]
    Returns:
        全局航向角（yaw）的角度值
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    yaw = np.arctan2(dy, dx) * 180 / np.pi
    return yaw


def get_valid_angle(angle):
    """
    将角度限制在 -180 到 180 的范围内
    Args:
        angle: 待限制的角度值
    Returns:
        限制在 -180 到 180 范围内的角度值
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < - np.pi:
        angle += 2 * np.pi
    return angle


def get_dist(x1, y1, x2, y2):
    """
    计算两点之间的欧几里德距离
    Args:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标
    Returns:
        两点之间的欧几里德距离
    """
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx ** 2 + dy ** 2)
    return distance

def calculate_steer_angle(ego, waypoint_list, k):
    v = get_speed(ego)
    yaw = np.radians(ego_transform.rotation.yaw)

    min_idx = 1
    for idx in range(len(waypoint_list)):
        dis = get_dist(ego_loc.x, ego_loc.y, waypoint_list[idx][0], waypoint_list[idx][1])
        if idx == 0:
            e_r = dis
        if dis < e_r:
            e_r = dis
            min_idx = idx

    trajectory_yaw = calculate_trajectory_yaw(waypoint_list, min_idx)
    heading_error = trajectory_yaw - yaw
    heading_error = get_valid_angle(heading_error)

    min_path_yaw = np.arctan2(waypoint_list[min_idx][1] - ego_loc.y,
                              waypoint_list[min_idx][0] - ego_loc.x)
    cross_yaw_error = min_path_yaw - yaw
    cross_yaw_error = get_valid_angle(cross_yaw_error)
    if cross_yaw_error > 0:
        e_r = e_r
    else:
        e_r = -e_r
    delta_error = np.arctan(k * e_r / (v + 1.0e-6))
    steer_angle = heading_error + delta_error

    return steer_angle


init_helper = InitHelper()

client = init_helper.connect2server()
world = init_helper.world
init_helper.set_world_settings()
spectator = init_helper.set_spectator_transform()
spawn_pts = init_helper.draw_spawn_points()
route = init_helper.global_path_planning(spawn_pts[132].location, spawn_pts[27].location)

wps = [waypoint[0] for waypoint in route]
init_helper.draw_waypoints(wps)

next = wps[0]
blueprint_library = world.get_blueprint_library()

# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
ego = world.spawn_actor(ego_bp, spawn_pts[132])
camera_helper = CameraHelper(world, ego)
camera_location = carla.Location(x=-5, z=3)
camera_rotation = carla.Rotation(pitch=-20)
camera = camera_helper.spawn_camera(camera_location, camera_rotation, camera_helper.show_image)

control = carla.VehicleControl()

waypoint_list = []

for wp in wps:
    waypoint_list.insert(wps.index(wp), (wp.transform.location.x, wp.transform.location.y))

pid = PIDLongitudinalController(ego, K_P=1, K_I=0.75, K_D=0.0, dt=0.01)

# Use stanley controller for lateral control

target_speed = 30

# Generate waypoints
i = 0

past_steering = 0

try:
    while True:
        ego_transform = ego.get_transform()
        ego_loc = ego.get_location()
        spectator.set_transform(
            carla.Transform(ego_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))

        world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=T)
        world.debug.draw_point(next.transform.location, color=carla.Color(g=255), life_time=T)
        ego_dist = distance_vehicle(next, ego_transform)

        steer_angle = calculate_steer_angle(ego, waypoint_list, 4.0)

        # 4. update

        control = carla.VehicleControl()
        throttle = pid.run_step(target_speed)

        if throttle >= 0.0:
            control.throttle = min(throttle, 0.5)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(throttle), 0.5)

        if steer_angle > past_steering + 0.1:
            steer_angle = past_steering + 0.1
        elif steer_angle < past_steering - 0.1:
            steer_angle = past_steering - 0.1

        if steer_angle >= 0:
            steering = min(0.8, steer_angle)
        else:
            steering = max(-0.8, steer_angle)

        control.steer = steering

        ego.apply_control(control)
        past_steering = steer_angle

        if i == (len(wps) - 1):
            control.brake = 1
            ego.apply_control(control)
            print('this trip finish')
            time.sleep(3)
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

