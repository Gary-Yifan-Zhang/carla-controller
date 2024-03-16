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

import carla
import numpy as np
import pygame
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import PIDLongitudinalController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector, get_speed

SHOW_CAM = True
distance = 2.0
T = 100

L = 2.875
Kdd = 4.0
alpha_prev = 0
delta_prev = 0


def show_image(image):
    # Convert the image.raw_data to a numpy array
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # Reshape the array to 4 channels per pixel (RGBA)
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    # Convert the image from RGBA to BGR (OpenCV uses BGR)
    image_array = image_array[:, :, :3]
    image_array = image_array[:, :, ::-1]
    # Display the image using OpenCV imshow()
    if SHOW_CAM:
        cv2.imshow("Camera", image_array)
        # Wait for a key press
        cv2.waitKey(1)
    # front_camera = image_array


def connect_to_server():
    """
    连接到Carla服务器并返回客户端对象
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(50.0)
        return client
    except Exception as e:
        print(f"An error occurred while connecting to the server: {e}")
        return None


def set_world_settings(world, synchronous_mode=False):
    """
    设置Carla世界的设置，包括固定的时间步和同步模式
    Args:
        world: Carla世界对象
        synchronous_mode: 是否启用同步模式，默认为True
    """
    original_settings = world.get_settings()

    try:
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.01
        settings.synchronous_mode = synchronous_mode
        world.apply_settings(settings)

        if synchronous_mode:
            traffic_manager = carla.Client('localhost', 2000).get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
    except Exception as e:
        print(f"Error occurred while setting the synchronization mode: {e}")
        # 恢复原始设置
        world.apply_settings(original_settings)


def set_spectator_transform(world):
    """
    设置观察者的位置和方向
    """
    try:
        spectator = world.get_spectator()
        transform = carla.Transform()
        bv_transform = carla.Transform(transform.location + carla.Location(z=200, x=0),
                                       carla.Rotation(yaw=0, pitch=-90))
        spectator.set_transform(bv_transform)
        return spectator
    except Exception as e:
        print(f"Error occurred while retrieving the list of spawn points：{e}")


def get_spawn_points(world):
    """
    获取生成点列表
    """
    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points()
        return spawn_points
    except Exception as e:
        print(f"获取生成点列表时出现错误：{e}")
        return None


def draw_spawn_points(world, spawn_points):
    """
    在Carla世界中绘制生成点
    """
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time=100)
        world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(),
                               life_time=100)


def global_path_planning(world, spawn_points, start_index, end_index):
    """
    全局路径规划
    """
    m = world.get_map()
    grp = GlobalRoutePlanner(m, distance)
    origin = carla.Location(spawn_points[start_index].location)
    destination = carla.Location(spawn_points[end_index].location)
    route = grp.trace_route(origin, destination)
    return route


def draw_waypoints(world, waypoints):
    """
    在Carla世界中绘制路点
    """
    T = 100
    for i in range(len(waypoints) - 1):
        pi_location = waypoints[i].transform.location
        pj_location = waypoints[i + 1].transform.location
        pi_location.z = 0.5
        pj_location.z = 0.5
        world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(b=255))
        pi_location.z = 0.6
        world.debug.draw_point(pi_location, color=carla.Color(b=255), life_time=T)


def set_synchronized_mode(world, client):
    """
    设置Carla世界和TrafficManager为同步模式
    """
    original_settings = world.get_settings()

    try:
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
    except Exception as e:
        print(f"Error occurred while setting the synchronization mode: {e}")
        # 恢复原始设置
        world.apply_settings(original_settings)


def spawn_camera(world, ego_vehicle, callback, x=0, y=0, z=0, pitch=0):
    """
    在指定位置和旋转角度上生成相机，并注册回调函数
    Args:
        world: Carla世界对象
        ego_vehicle: 车辆对象，将相机附加到该车辆上
        location: 相机位置的carla.Location对象
        rotation: 相机旋转角度的carla.Rotation对象
        callback: 相机数据回调函数
    Returns:
        相机对象
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_location = carla.Location(x=x, z=z)
    camera_rotation = carla.Rotation(pitch=pitch)
    camera_trans = carla.Transform(camera_location, camera_rotation)
    camera = world.spawn_actor(camera_bp, camera_trans, attach_to=ego_vehicle)
    camera.listen(callback)
    return camera


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


def calc_steering_angle(alpha, ld):
    delta_prev = 0
    delta = math.atan2(2 * L * np.sin(alpha), ld)
    delta = np.fmax(np.fmin(delta, 1.0), -1.0)
    if math.isnan(delta):
        delta = delta_prev
    else:
        delta_prev = delta

    return delta


def get_lookahead_dist(vf, idx, waypoint_list, dist):
    ld = Kdd * vf
    # while ld > dist[idx] and (idx+1) < len(waypoint_list):
    #     idx += 1
    return ld


client = connect_to_server()
if client:
    world = client.get_world()
    if world:
        set_world_settings(world, synchronous_mode=False)
        spectator = set_spectator_transform(world)
        spawn_points = get_spawn_points(world)
        # set_synchronized_mode(world, client)
        if spawn_points:
            draw_spawn_points(world, spawn_points)
            route = global_path_planning(world, spawn_points, 88, 27)
            if route:
                wps = [waypoint[0] for waypoint in route]
                draw_waypoints(world, wps)

next = wps[0]
blueprint_library = world.get_blueprint_library()

# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
ego = world.spawn_actor(ego_bp, spawn_points[88])
camera = spawn_camera(world, ego, show_image, x=-5, y=0, z=3, pitch=-20)

control = carla.VehicleControl()

waypoint_list = []

for wp in wps:
    waypoint_list.insert(wps.index(wp), (wp.transform.location.x, wp.transform.location.y))

pid = PIDLongitudinalController(ego, K_P=1, K_I=0.75, K_D=0.0, dt=0.01)

target_speed = 30

# Generate waypoints
i = 0
past_steering = 0
try:
    while True:
        ego_transform = ego.get_transform()
        spectator.set_transform(
            carla.Transform(ego_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))

        ego_loc = ego.get_location()
        world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=T)
        world.debug.draw_point(next.transform.location, color=carla.Color(r=255), life_time=T)
        ego_dist = distance_vehicle(next, ego_transform)
        ego_vel = ego.get_velocity()
        # print(ego_vel)

        vf = np.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)
        vf = np.fmax(np.fmin(vf, 2.5), 0.1)

        min_index, tx, ty, dist = get_target_wp_index(ego_loc, waypoint_list)
        ld = get_lookahead_dist(vf, min_index, waypoint_list, dist)

        yaw = np.radians(ego_transform.rotation.yaw)
        alpha = math.atan2(ty - ego_loc.y, tx - ego_loc.x) - yaw
        # alpha = np.arccos((ex*np.cos(yaw)+ey*np.sin(yaw))/ld)

        if math.isnan(alpha):
            alpha = alpha_prev
        else:
            alpha_prev = alpha

        e = np.sin(alpha) * ld

        throttle = pid.run_step(target_speed)
        control = carla.VehicleControl()
        if throttle >= 0.0:
            control.throttle = min(throttle, 0.5)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(throttle), 0.3)

        steer_angle = calc_steering_angle(alpha, ld)
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

        past_steering = steering

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
