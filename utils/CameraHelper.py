import carla
import numpy as np
import cv2
class CameraHelper:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle

    def spawn_camera(self, location, rotation, callback):
        """
        在指定位置和旋转角度上生成相机，并注册回调函数
        Args:
            location: 相机位置的carla.Location对象
            rotation: 相机旋转角度的carla.Rotation对象
            callback: 相机数据回调函数
        Returns:
            相机对象
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_trans = carla.Transform(location, rotation)
        camera = self.world.spawn_actor(camera_bp, camera_trans, attach_to=self.ego_vehicle)
        camera.listen(callback)
        return camera

    def show_image(self, image):
        """
        显示相机图像
        Args:
            image: 相机图像数据
        """
        image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (image.height, image.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]

        cv2.imshow("Camera", image_array)
        cv2.waitKey(1)