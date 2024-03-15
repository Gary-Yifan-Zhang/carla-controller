import glob
import os
import sys
import cv2

sys.path.append("D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla")
sys.path.append("D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\agents")

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import pygame
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector


class InitHelper:
    def __init__(self, host='localhost', port=2000, timeout=10):
        self.distance = 2.0
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()

    def connect_to_server(self):
        """
        连接到Carla服务器并返回客户端对象
        """
        return self.client

    def set_world_settings(self, synchronous_mode=False):
        """
        设置Carla世界的设置，包括固定的时间步和同步模式
        Args:
            synchronous_mode: 是否启用同步模式，默认为True
        """
        original_settings = self.world.get_settings()

        try:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.01
            settings.synchronous_mode = synchronous_mode
            self.world.apply_settings(settings)

            if synchronous_mode:
                traffic_manager = self.client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)
        except Exception as e:
            print(f"设置同步模式时出现错误：{e}")
            # 恢复原始设置
            self.world.apply_settings(original_settings)

    def set_spectator_transform(self):
        """
        设置观察者的位置和方向
        """
        try:
            spectator = self.world.get_spectator()
            transform = carla.Transform()
            bv_transform = carla.Transform(transform.location + carla.Location(z=200, x=0),
                                           carla.Rotation(yaw=0, pitch=-90))
            spectator.set_transform(bv_transform)
            return spectator
        except Exception as e:
            print(f"设置观察者位置和方向时出现错误：{e}")

    def draw_spawn_points(self):
        """
        获取生成点列表
        在Carla世界中绘制生成点
        """
        m = self.world.get_map()
        spawn_points = m.get_spawn_points()
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=100)
            self.world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(),
                                        life_time=100)

        return spawn_points

    def global_path_planning(self, start_index, end_index):
        """
        全局路径规划
        """
        m = self.world.get_map()
        grp = GlobalRoutePlanner(m, self.distance)
        spawn_points = self.draw_spawn_points()  # 获取生成点列表
        origin = carla.Location(spawn_points[start_index].location)
        destination = carla.Location(spawn_points[end_index].location)
        route = grp.trace_route(origin, destination)
        return route

    def draw_waypoints(self, waypoints):
        """
        在Carla世界中绘制路点
        """
        T = 100
        for i in range(len(waypoints) - 1):
            pi_location = waypoints[i].transform.location
            pj_location = waypoints[i + 1].transform.location
            pi_location.z = 0.5
            pj_location.z = 0.5
            self.world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(b=255))
            pi_location.z = 0.6
            self.world.debug.draw_point(pi_location, color=carla.Color(b=255), life_time=T)

    def set_synchronized_mode(self):
        """
        设置Carla世界和TrafficManager为同步模式
        """
        original_settings = self.world.get_settings()

        try:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
        except Exception as e:
            print(f"设置同步模式时出现错误：{e}")
            # 恢复原始设置
            self.world.apply_settings(original_settings)

    def spawn_camera(self, location, rotation, callback):
        """
        在指定位置和旋转角度上生成相机，并注册回调函数
        Args:
            location: 相机位置的carla.Location对象
            rotation: 相机旋转角度的carla.Rotation对象
            callback: 相机数据回调函数
        Returns:
            相机对象：
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_trans = carla.Transform(location, rotation)
        camera = self.world.spawn_actor(camera_bp, camera_trans, attach_to=self.ego_vehicle)
        camera.listen(callback)
        return camera
