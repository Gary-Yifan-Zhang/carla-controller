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

SHOW_CAM = True
distance = 2.0
T = 100

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
        print(f"连接服务器时出现错误：{e}")
        return None


def set_world_settings(world, synchronous_mode=True):
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
        print(f"设置同步模式时出现错误：{e}")
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
        print(f"设置观察者位置和方向时出现错误：{e}")


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
        print(f"设置同步模式时出现错误：{e}")
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

blueprint_library = world.get_blueprint_library()

# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
ego = world.spawn_actor(ego_bp, spawn_points[88])
camera = spawn_camera(world, ego, show_image, x=-5, y=0, z=3, pitch=-20)


# PID
# args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}
args_lateral_dict = {'K_P': 1.95, 'K_D': 0.2, 'K_I': 0.07, 'dt': 1.0 / 10.0}

# args_long_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05}
args_long_dict = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.75, 'dt': 1.0 / 10.0}

PID = VehiclePIDController(ego, args_lateral=args_lateral_dict, args_longitudinal=args_long_dict)

i = 0
target_speed = 30
next = wps[0]




try:
    while True:
        ego_transform = ego.get_transform()
        spectator.set_transform(
            carla.Transform(ego_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))

        ego_loc = ego.get_location()
        world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=T)
        world.debug.draw_point(next.transform.location, color=carla.Color(r=255), life_time=T)
        ego_dist = distance_vehicle(next, ego_transform)
        ego_vect = vector(ego_loc, next.transform.location)

        control = PID.run_step(target_speed, next)

        if i == (len(wps) - 1):
            control = PID.run_step(0, wps[-1])
            ego.apply_control(control)
            print('this trip finish')
            break

        if ego_dist < 1.5:
            i = i + 1
            next = wps[i]
            control = PID.run_step(target_speed, next)

        world.wait_for_tick()
        ego.apply_control(control)

        # # Update the display
        # gameDisplay.blit(renderObject.surface, (0, 0))
        # pygame.display.flip()


finally:
    ego.destroy()
    camera.stop()
    pygame.quit()
