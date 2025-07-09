#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时点云姿态估计与STL模型可视化程序
功能：
- 实时获取点云数据
- 调用粗配准和ICP精配准算法
- 实时展示匹配结果的STL模型
- 按'q'键退出程序
"""

import sys
import os
import numpy as np
import torch
import copy
import time
import datetime
from scipy.spatial.transform import Rotation as R

# Mecheye相机相关导入
try:
    from mecheye.shared import *
    from mecheye.area_scan_3d_camera import *
    from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d
    MECHEYE_AVAILABLE = True
except ImportError:
    print("警告: Mecheye SDK未找到，将使用虚拟数据")
    MECHEYE_AVAILABLE = False

# Open3D导入
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("错误: Open3D未找到，请安装Open3D (pip install open3d)")
    sys.exit(1)

# PyTorch相关导入
try:
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    print("错误: PyTorch未找到，请安装PyTorch")
    print("安装命令: pip install torch")
    sys.exit(1)

# YAML配置文件支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("警告: PyYAML未找到，将使用默认参数")
    YAML_AVAILABLE = False

class RealtimePoseEstimation:
    def __init__(self, config_file="realtime_pose_config.yaml"):
        self.camera = Camera() if MECHEYE_AVAILABLE else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        self.config = self.load_config(config_file)
        
        # 设置参数
        self.model_file = self.config.get('model_file', "stp/part2_rude.STL")
        
        # 处理参数
        processing = self.config.get('processing', {})
        self.model_sample_points = processing.get('model_sample_points', 20480)
        self.preprocess_voxel_size = processing.get('preprocess_voxel_size', 0.002)
        self.preprocess_sor_k = processing.get('preprocess_sor_k', 20)
        self.preprocess_sor_std_ratio = processing.get('preprocess_sor_std_ratio', 2.0)
        self.preprocess_fps_n_points = processing.get('preprocess_fps_n_points', 2048)
        
        # 粗配准参数
        coarse = self.config.get('coarse_alignment', {})
        self.coarse_iterations = coarse.get('iterations', 200)
        self.coarse_lr = coarse.get('learning_rate', 0.01)
        self.coarse_align_scene_points = coarse.get('scene_points', 2048)
        self.coarse_align_model_points = coarse.get('model_points', 2048)
        self.coarse_print_every = coarse.get('print_every', 20)
        
        # ICP参数
        icp = self.config.get('icp', {})
        self.icp_threshold = icp.get('threshold', 30.0)
        self.icp_max_iter = icp.get('max_iterations', 5000)
        self.icp_relative_rmse = icp.get('relative_rmse', 0.00001)
        self.icp_relative_fitness = icp.get('relative_fitness', 0.00001)
        self.icp_transform_change_threshold = icp.get('transform_change_threshold', 0.001)
        self.icp_min_points = icp.get('min_points', 100)
        
        # 实时参数
        realtime = self.config.get('realtime', {})
        self.pose_interval = realtime.get('pose_estimation_interval', 2.0)
        self.frame_delay = realtime.get('frame_rate_delay', 0.05)
        
        # 可视化参数
        visualization = self.config.get('visualization', {})
        self.window_width = visualization.get('window_width', 1280)
        self.window_height = visualization.get('window_height', 960)
        self.point_size = visualization.get('point_size', 2.0)
        self.show_coordinate_frame = visualization.get('show_coordinate_frame', True)
        self.background_color = visualization.get('background_color', [0.1, 0.1, 0.1])
        
        # 显示控制选项
        display = visualization.get('display', {})
        self.show_original_scene = display.get('show_original_scene', True)
        self.show_preprocessed_scene = display.get('show_preprocessed_scene', True)
        self.show_target_model = display.get('show_target_model', True)
        self.show_matched_model = display.get('show_matched_model', True)
        self.show_model_pointcloud = display.get('show_model_pointcloud', False)
        self.show_coarse_alignment = display.get('show_coarse_alignment', False)
        self.show_correspondences = display.get('show_correspondences', False)
        self.show_bounding_boxes = display.get('show_bounding_boxes', False)
        self.show_center_points = display.get('show_center_points', False)
        
        # 颜色配置
        colors = visualization.get('colors', {})
        self.original_scene_color = colors.get('original_scene', [0.0, 0.0, 1.0])
        self.preprocessed_scene_color = colors.get('preprocessed_scene', [0.0, 0.8, 0.8])
        self.target_model_color = colors.get('target_model', [0.0, 1.0, 0.0])
        self.matched_model_color = colors.get('matched_model', [1.0, 0.0, 0.0])
        self.model_pointcloud_color = colors.get('model_pointcloud', [1.0, 1.0, 0.0])
        self.coarse_alignment_color = colors.get('coarse_alignment', [1.0, 0.5, 0.0])
        self.correspondences_color = colors.get('correspondences', [0.8, 0.8, 0.8])
        self.bounding_box_color = colors.get('bounding_box', [0.5, 0.5, 0.5])
        self.center_point_color = colors.get('center_point', [1.0, 0.0, 1.0])
        
        # 数据保存选项
        data_saving = self.config.get('data_saving', {})
        self.enable_saving = data_saving.get('enable_saving', False)
        self.save_directory = data_saving.get('save_directory', "captured_data")
        self.save_interval = data_saving.get('save_interval', 10)
        self.save_formats = data_saving.get('save_formats', {})
        self.save_items = data_saving.get('save_items', {})
        
        # 2D图像显示选项
        image_display = self.config.get('image_display', {})
        self.enable_2d_display = image_display.get('enable_2d_display', False)
        self.image_window_position = image_display.get('window_position', [100, 100])
        self.image_window_size = image_display.get('window_size', [640, 480])
        self.image_display_items = image_display.get('display_items', {})
        
        # 相机参数
        camera_config = self.config.get('camera', {})
        self.connection_timeout = camera_config.get('connection_timeout', 10.0)
        self.capture_retry_count = camera_config.get('capture_retry_count', 3)
        self.capture_timeout = camera_config.get('capture_timeout', 5.0)
        
        # 初始化存储目录
        if self.enable_saving:
            self.setup_save_directory()
        
        # 加载目标模型
        self.target_mesh_original = None
        self.target_pcd_for_matching = None
        self.load_target_model()
        
        # 可视化器
        self.vis = None
        self.setup_visualizer()
        
        # 2D图像显示窗口
        if self.enable_2d_display:
            try:
                import cv2
                self.cv2 = cv2
                self.image_windows_initialized = False
            except ImportError:
                print("警告: OpenCV未安装，无法显示2D图像")
                self.enable_2d_display = False
        
        # 运行时数据存储
        self.current_frame_data = {}
        self.frame_counter = 0
        self.coarse_transform_result = None
        
        print(f"使用设备: {self.device}")
        print(f"模型文件: {self.model_file}")
        print(f"数据保存: {'启用' if self.enable_saving else '禁用'}")
        print(f"2D图像显示: {'启用' if self.enable_2d_display else '禁用'}")
        
    def setup_save_directory(self):
        """设置保存目录"""
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                print(f"创建保存目录: {self.save_directory}")
            
            # 创建子目录
            subdirs = ['pointclouds', 'images', 'poses', 'results']
            for subdir in subdirs:
                subdir_path = os.path.join(self.save_directory, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                    
        except Exception as e:
            print(f"创建保存目录失败: {e}")
            self.enable_saving = False
        
    def load_config(self, config_file):
        """加载配置文件"""
        if not YAML_AVAILABLE or not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在或YAML不可用，使用默认参数")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"成功加载配置文件: {config_file}")
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认参数")
            return {}
        
    def load_target_model(self):
        """加载目标STL模型"""
        try:
            print(f"加载目标模型: {self.model_file}")
            
            # 加载STL文件
            self.target_mesh_original = o3d.io.read_triangle_mesh(self.model_file)
            if not self.target_mesh_original.has_vertices():
                raise ValueError("STL文件加载失败或为空")
            
            # 从网格采样点云用于匹配
            self.target_pcd_for_matching = self.target_mesh_original.sample_points_uniformly(
                number_of_points=self.model_sample_points)
            
            if not self.target_mesh_original.has_vertex_colors():
                self.target_mesh_original.paint_uniform_color(self.target_model_color)
            
            if not self.target_mesh_original.has_vertex_normals():
                self.target_mesh_original.compute_vertex_normals()
            
            print(f"目标模型加载成功，顶点数: {len(self.target_mesh_original.vertices)}")
            print(f"采样点云数: {len(self.target_pcd_for_matching.points)}")
            
        except Exception as e:
            print(f"加载目标模型失败: {e}")
            sys.exit(1)
    
    def setup_visualizer(self):
        """设置Open3D可视化器"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="实时姿态估计 - 关闭窗口退出", 
                width=self.window_width, 
                height=self.window_height
            )
            
            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.show_coordinate_frame = self.show_coordinate_frame
            render_option.background_color = self.background_color
            render_option.point_size = self.point_size
            
            # 添加一个初始的坐标系以确保窗口有内容
            initial_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(initial_coord_frame, reset_bounding_box=True)
            
            # 设置初始视角
            view_control = self.vis.get_view_control()
            view_control.set_zoom(0.7)
            view_control.set_front([0.5, 0.5, 0.5])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 0, 1])
            
            # 更新一次渲染
            self.vis.poll_events()
            self.vis.update_renderer()
            
            print("Open3D可视化窗口已创建并初始化")
            print("如果看不到窗口，请检查任务栏或将其置于前台")
            
            # 设置键盘退出标志
            self.should_exit = False
            
            # 注册键盘回调
            def key_callback(vis, action, mods):
                if action == 1:  # 按键按下
                    self.should_exit = True
                    print("检测到键盘输入，程序将退出...")
                return False
            
            # 由于Open3D的键盘回调在某些情况下可能不稳定，我们将依赖窗口关闭检测
            
        except Exception as e:
            print(f"设置可视化器失败: {e}")
            sys.exit(1)
    
    def connect_camera(self):
        """连接相机"""
        if not MECHEYE_AVAILABLE:
            print("错误: Mecheye SDK不可用，无法连接相机")
            print("请安装Mecheye Python SDK才能使用相机功能")
            return False
            
        try:
            if find_and_connect(self.camera):
                print("相机连接成功")
                return True
            else:
                print("错误：未找到可用的相机")
                print("请检查：")
                print("1. 相机是否已连接")
                print("2. 相机是否已开机")
                print("3. 网络连接是否正常")
                return False
        except Exception as e:
            print(f"连接相机时出错: {e}")
            return False
    
    def capture_point_cloud_from_camera(self):
        """从相机捕获点云和图像数据"""
        if not MECHEYE_AVAILABLE:
            print("错误: Mecheye SDK未安装，无法使用相机")
            return None, None, None
            
        if self.camera is None:
            print("错误: 相机未连接")
            return None, None, None
            
        # 尝试多次捕获，提高成功率
        for attempt in range(self.capture_retry_count):
            try:
                if attempt > 0:
                    print(f"重试捕获数据 (第{attempt + 1}次尝试)...")
                
                # 捕获2D和3D数据
                frame_2d_and_3d = Frame2DAnd3D()
                show_error(self.camera.capture_2d_and_3d(frame_2d_and_3d))
                
                # 获取3D和2D帧数据
                frame_3d = frame_2d_and_3d.frame_3d()
                frame_2d = frame_2d_and_3d.frame_2d()
                
                # 获取点云
                temp_pcd_file = f"temp_pointcloud_{attempt}.ply"
                pcd = None
                try:
                    # 保存点云到临时文件
                    show_error(frame_3d.save_untextured_point_cloud(FileFormat_PLY, temp_pcd_file))
                    
                    # 用Open3D读取临时文件
                    pcd = o3d.io.read_point_cloud(temp_pcd_file)
                    
                    # 删除临时文件
                    if os.path.exists(temp_pcd_file):
                        os.remove(temp_pcd_file)
                        
                except Exception as e_pcd:
                    print(f"处理点云时出错: {e_pcd}")
                    if os.path.exists(temp_pcd_file):
                        os.remove(temp_pcd_file)
                
                # 获取RGB图像 - 参考realtime_rgb_with_pointcloud_capture.py的实现
                rgb_image = None
                try:
                    # 检查颜色类型并获取对应图像
                    if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
                        image2d = frame_2d.get_gray_scale_image()
                        # 转换为numpy数组并转换为彩色
                        gray_array = np.array(image2d.data(), copy=False)
                        if len(gray_array.shape) == 2:
                            # 灰度图转换为RGB
                            rgb_image = np.stack([gray_array, gray_array, gray_array], axis=2)
                        else:
                            rgb_image = gray_array
                    elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
                        image2d = frame_2d.get_color_image()
                        # 转换为numpy数组
                        rgb_image = np.array(image2d.data(), copy=False)
                    else:
                        print("警告：未知的图像颜色类型")
                        
                except Exception as e_rgb:
                    print(f"获取RGB图像时出错: {e_rgb}")
                
                # 获取深度图像 - 直接尝试获取深度图
                depth_image = None
                try:
                    # 尝试获取深度图，如果不存在会抛出异常
                    depth_map = frame_3d.get_depth_map()
                    # 转换为numpy数组
                    depth_image = np.array(depth_map.data(), copy=False)
                except Exception as e_depth:
                    print(f"获取深度图像时出错: {e_depth}")
                    # 深度图可能不可用，这是正常的
                
                # 检查是否至少获取到点云
                if pcd is not None and pcd.has_points():
                    print(f"成功捕获数据 - 点云: {len(pcd.points)}点, RGB: {'✓' if rgb_image is not None else '✗'}, 深度: {'✓' if depth_image is not None else '✗'}")
                    return pcd, rgb_image, depth_image
                else:
                    print("警告: 捕获的点云为空")
                    if attempt < self.capture_retry_count - 1:
                        time.sleep(0.5)
                        continue
                        
            except Exception as e:
                print(f"捕获数据时出错: {e}")
                if attempt < self.capture_retry_count - 1:
                    time.sleep(0.5)
                    continue
        
        return None, None, None
        
    def save_frame_data(self, frame_num, scene_pcd=None, preprocessed_pcd=None, 
                       rgb_image=None, depth_image=None, pose_matrix=None, 
                       matching_result=None):
        """保存帧数据"""
        if not self.enable_saving or frame_num % self.save_interval != 0:
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_id = f"frame_{frame_num:06d}_{timestamp}"
            
            # 保存点云
            if self.save_items.get('original_pointcloud', False) and scene_pcd is not None:
                for fmt in self.save_formats.get('pointcloud', ['ply']):
                    filename = os.path.join(self.save_directory, 'pointclouds', 
                                          f"{frame_id}_original.{fmt}")
                    o3d.io.write_point_cloud(filename, scene_pcd)
            
            if self.save_items.get('preprocessed_pointcloud', False) and preprocessed_pcd is not None:
                for fmt in self.save_formats.get('pointcloud', ['ply']):
                    filename = os.path.join(self.save_directory, 'pointclouds', 
                                          f"{frame_id}_preprocessed.{fmt}")
                    o3d.io.write_point_cloud(filename, preprocessed_pcd)
            
            # 保存RGB图像
            if self.save_items.get('rgb_image', False) and rgb_image is not None:
                for fmt in self.save_formats.get('image', ['png']):
                    filename = os.path.join(self.save_directory, 'images', 
                                          f"{frame_id}_rgb.{fmt}")
                    try:
                        if self.enable_2d_display and hasattr(self, 'cv2'):
                            # 确保图像格式正确
                            if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                                # RGB转BGR（OpenCV格式）
                                bgr_image = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR)
                                self.cv2.imwrite(filename, bgr_image)
                            else:
                                # 直接保存灰度或其他格式
                                self.cv2.imwrite(filename, rgb_image)
                        else:
                            # 如果没有OpenCV，尝试其他方法保存
                            print(f"警告: 无法保存RGB图像 {filename}，需要OpenCV")
                    except Exception as e_save:
                        print(f"保存RGB图像失败 {filename}: {e_save}")
            
            # 保存深度图像
            if self.save_items.get('depth_image', False) and depth_image is not None:
                for fmt in self.save_formats.get('image', ['png']):
                    filename = os.path.join(self.save_directory, 'images', 
                                          f"{frame_id}_depth.{fmt}")
                    try:
                        if self.enable_2d_display and hasattr(self, 'cv2'):
                            # 归一化深度图像用于保存
                            if depth_image.max() > 0:
                                depth_normalized = (depth_image / depth_image.max() * 255).astype('uint8')
                                self.cv2.imwrite(filename, depth_normalized)
                            else:
                                print(f"警告: 深度图像数据异常，跳过保存 {filename}")
                        else:
                            print(f"警告: 无法保存深度图像 {filename}，需要OpenCV")
                    except Exception as e_save:
                        print(f"保存深度图像失败 {filename}: {e_save}")
            
            # 保存姿态矩阵
            if self.save_items.get('pose_matrix', False) and pose_matrix is not None:
                if 'txt' in self.save_formats.get('pose', ['txt']):
                    filename = os.path.join(self.save_directory, 'poses', 
                                          f"{frame_id}_pose.txt")
                    np.savetxt(filename, pose_matrix)
                
                if 'json' in self.save_formats.get('pose', ['json']):
                    filename = os.path.join(self.save_directory, 'poses', 
                                          f"{frame_id}_pose.json")
                    import json
                    pose_data = {
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'transformation_matrix': pose_matrix.tolist()
                    }
                    with open(filename, 'w') as f:
                        json.dump(pose_data, f, indent=2)
            
            # 保存匹配结果
            if self.save_items.get('matching_result', False) and matching_result is not None:
                filename = os.path.join(self.save_directory, 'results', 
                                      f"{frame_id}_result.json")
                import json
                with open(filename, 'w') as f:
                    json.dump(matching_result, f, indent=2)
                    
            print(f"保存帧 {frame_num} 数据完成")
            
        except Exception as e:
            print(f"保存帧数据失败: {e}")
    
    def display_2d_images(self, rgb_image=None, depth_image=None):
        """显示2D图像"""
        if not self.enable_2d_display or not hasattr(self, 'cv2'):
            return
            
        try:
            # 初始化窗口
            if not self.image_windows_initialized:
                if rgb_image is not None and self.image_display_items.get('rgb_image', True):
                    self.cv2.namedWindow('RGB Image', self.cv2.WINDOW_NORMAL)
                    self.cv2.resizeWindow('RGB Image', self.image_window_size[0], self.image_window_size[1])
                    
                if depth_image is not None and self.image_display_items.get('depth_image', True):
                    self.cv2.namedWindow('Depth Image', self.cv2.WINDOW_NORMAL)
                    self.cv2.resizeWindow('Depth Image', self.image_window_size[0], self.image_window_size[1])
                
                self.image_windows_initialized = True
            
            # 显示RGB图像
            if rgb_image is not None and self.image_display_items.get('rgb_image', True):
                try:
                    # 确保图像格式正确
                    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                        # 从RGB转换为BGR (OpenCV使用BGR)
                        rgb_display = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR)
                        self.cv2.imshow('RGB Image', rgb_display)
                    elif len(rgb_image.shape) == 2:
                        # 灰度图像直接显示
                        self.cv2.imshow('RGB Image', rgb_image)
                    else:
                        # 其他格式，尝试直接显示
                        self.cv2.imshow('RGB Image', rgb_image)
                except Exception as e_rgb_display:
                    print(f"显示RGB图像失败: {e_rgb_display}")
            
            # 显示深度图像
            if depth_image is not None and self.image_display_items.get('depth_image', True):
                try:
                    # 归一化深度图像用于显示
                    if depth_image.max() > 0:
                        depth_normalized = (depth_image / depth_image.max() * 255).astype('uint8')
                        depth_colored = self.cv2.applyColorMap(depth_normalized, self.cv2.COLORMAP_JET)
                        self.cv2.imshow('Depth Image', depth_colored)
                    else:
                        print("警告: 深度图像数据异常，无法显示")
                except Exception as e_depth_display:
                    print(f"显示深度图像失败: {e_depth_display}")
            
            # 处理键盘事件
            key = self.cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q'键或ESC键
                self.should_exit = True
                
        except Exception as e:
            print(f"显示2D图像失败: {e}")
    
    def create_bounding_box_geometry(self, geometry):
        """创建边界框几何体"""
        try:
            bbox = geometry.get_axis_aligned_bounding_box()
            bbox.color = self.bounding_box_color
            return bbox
        except Exception as e:
            print(f"创建边界框失败: {e}")
            return None
    
    def create_center_point_geometry(self, geometry, radius=0.005):
        """创建中心点几何体"""
        try:
            center = geometry.get_center()
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(center)
            sphere.paint_uniform_color(self.center_point_color)
            return sphere
        except Exception as e:
            print(f"创建中心点失败: {e}")
            return None
    
    def create_correspondences_geometry(self, source_points, target_points, max_distance=None):
        """创建对应点连线几何体"""
        try:
            if max_distance is None:
                max_distance = self.icp_threshold
                
            lines = []
            colors = []
            
            # 找到对应点
            dists = np.linalg.norm(source_points[:, np.newaxis] - target_points, axis=2)
            min_indices = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)
            
            # 只显示距离小于阈值的对应点
            valid_mask = min_dists < max_distance
            valid_source = source_points[valid_mask]
            valid_target = target_points[min_indices[valid_mask]]
            
            if len(valid_source) == 0:
                return None
            
            # 创建连线
            all_points = np.vstack([valid_source, valid_target])
            for i in range(len(valid_source)):
                lines.append([i, i + len(valid_source)])
                # 根据距离设置颜色强度
                intensity = 1.0 - min(min_dists[valid_mask][i] / max_distance, 1.0)
                color = [c * intensity for c in self.correspondences_color]
                colors.append(color)
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(all_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            return line_set
            
        except Exception as e:
            print(f"创建对应点连线失败: {e}")
            return None
    
    def create_coordinate_frame(self, transform=None, size=0.05):
        """创建坐标系几何体"""
        try:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            if transform is not None:
                coord_frame.transform(transform)
            return coord_frame
        except Exception as e:
            print(f"创建坐标系失败: {e}")
            return None
    
    def update_visualization(self, scene_pcd=None, preprocessed_pcd=None, estimated_pose=None, 
                           coarse_transform=None, matching_info=None):
        """更新可视化"""
        try:
            # 清除之前的几何体
            self.vis.clear_geometries()
            
            geometries_to_add = []
            all_points = []  # 用于计算整体边界框
            
            # 1. 显示原始场景点云
            if self.show_original_scene and scene_pcd is not None and scene_pcd.has_points():
                scene_vis = copy.deepcopy(scene_pcd)
                if not scene_vis.has_colors():
                    scene_vis.paint_uniform_color(self.original_scene_color)
                geometries_to_add.append(("原始场景点云", scene_vis))
                all_points.extend(np.asarray(scene_vis.points))
                
                # 边界框
                if self.show_bounding_boxes:
                    bbox = self.create_bounding_box_geometry(scene_vis)
                    if bbox is not None:
                        geometries_to_add.append(("场景边界框", bbox))
                
                # 中心点
                if self.show_center_points:
                    center = self.create_center_point_geometry(scene_vis)
                    if center is not None:
                        geometries_to_add.append(("场景中心点", center))
            
            # 2. 显示预处理后的场景点云
            if self.show_preprocessed_scene and preprocessed_pcd is not None and preprocessed_pcd.has_points():
                preprocessed_vis = copy.deepcopy(preprocessed_pcd)
                if not preprocessed_vis.has_colors():
                    preprocessed_vis.paint_uniform_color(self.preprocessed_scene_color)
                geometries_to_add.append(("预处理场景点云", preprocessed_vis))
                all_points.extend(np.asarray(preprocessed_vis.points))
            
            # 3. 显示目标模型
            if self.show_target_model and self.target_mesh_original is not None:
                target_vis = copy.deepcopy(self.target_mesh_original)
                if not target_vis.has_vertex_colors():
                    target_vis.paint_uniform_color(self.target_model_color)
                geometries_to_add.append(("目标模型", target_vis))
                if target_vis.has_vertices():
                    all_points.extend(np.asarray(target_vis.vertices))
                
                # 边界框
                if self.show_bounding_boxes:
                    bbox = self.create_bounding_box_geometry(target_vis)
                    if bbox is not None:
                        geometries_to_add.append(("目标模型边界框", bbox))
            
            # 4. 显示匹配后的模型
            if self.show_matched_model and estimated_pose is not None and self.target_mesh_original is not None:
                matched_vis = copy.deepcopy(self.target_mesh_original)
                matched_vis.transform(estimated_pose)
                matched_vis.paint_uniform_color(self.matched_model_color)
                geometries_to_add.append(("匹配后模型", matched_vis))
                if matched_vis.has_vertices():
                    all_points.extend(np.asarray(matched_vis.vertices))
                
                # 坐标系
                if self.show_coordinate_frame:
                    coord_frame = self.create_coordinate_frame(estimated_pose)
                    if coord_frame is not None:
                        geometries_to_add.append(("匹配坐标系", coord_frame))
            
            # 5. 显示模型采样点云
            if self.show_model_pointcloud and self.target_pcd_for_matching is not None:
                model_pcd_vis = copy.deepcopy(self.target_pcd_for_matching)
                model_pcd_vis.paint_uniform_color(self.model_pointcloud_color)
                geometries_to_add.append(("模型采样点云", model_pcd_vis))
                all_points.extend(np.asarray(model_pcd_vis.points))
            
            # 6. 显示粗配准结果
            if self.show_coarse_alignment and coarse_transform is not None and self.target_mesh_original is not None:
                coarse_vis = copy.deepcopy(self.target_mesh_original)
                coarse_vis.transform(coarse_transform)
                coarse_vis.paint_uniform_color(self.coarse_alignment_color)
                geometries_to_add.append(("粗配准结果", coarse_vis))
                if coarse_vis.has_vertices():
                    all_points.extend(np.asarray(coarse_vis.vertices))
            
            # 7. 显示对应点连线
            if (self.show_correspondences and preprocessed_pcd is not None and 
                self.target_pcd_for_matching is not None and matching_info is not None):
                correspondences = self.create_correspondences_geometry(
                    np.asarray(preprocessed_pcd.points),
                    np.asarray(self.target_pcd_for_matching.points)
                )
                if correspondences is not None:
                    geometries_to_add.append(("对应点连线", correspondences))
            
            # 添加所有几何体到可视化器
            first_geometry = True
            for name, geometry in geometries_to_add:
                self.vis.add_geometry(geometry, reset_bounding_box=first_geometry)
                first_geometry = False
            
            # 如果有几何体，设置视角
            if len(geometries_to_add) > 0 and len(all_points) > 0:
                # 计算所有点的边界
                all_points = np.array(all_points)
                bbox_min = np.min(all_points, axis=0)
                bbox_max = np.max(all_points, axis=0)
                center = (bbox_min + bbox_max) / 2
                size = np.max(bbox_max - bbox_min)
                
                # 设置视图控制器
                view_control = self.vis.get_view_control()
                view_control.set_lookat(center)  # 设置观察中心
                
                # 根据尺寸调整视角距离
                if size > 100:  # 毫米尺度
                    view_control.set_zoom(0.3)
                    view_control.set_front([0.3, 0.3, 0.6])
                elif size > 10:  # 厘米尺度
                    view_control.set_zoom(0.5)
                    view_control.set_front([0.4, 0.4, 0.6])
                else:  # 米尺度或更小
                    view_control.set_zoom(0.8)
                    view_control.set_front([0.5, 0.5, 0.5])
                
                view_control.set_up([0, 0, 1])  # 设置上向量
                
                print(f"数据边界: {bbox_min} 到 {bbox_max}, 中心: {center}, 尺寸: {size}")
            
            # 更新可视化
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # 输出显示信息
            if len(geometries_to_add) > 0:
                displayed_items = [name for name, _ in geometries_to_add]
                print(f"显示项目: {', '.join(displayed_items)}")
                print(f"可视化窗口已更新，几何体数量: {len(geometries_to_add)}")
            else:
                print("警告: 没有可显示的几何体")
            
        except Exception as e:
            print(f"更新可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_virtual_point_cloud(self):
        """生成虚拟点云数据进行测试"""
        # 从目标模型采样点云并添加变换和噪声来模拟观测数据
        if self.target_pcd_for_matching is not None and self.target_pcd_for_matching.has_points():
            # 从目标模型获取点云
            target_points = np.asarray(self.target_pcd_for_matching.points)
            
            # 随机选择一部分点来模拟部分观测
            n_points = min(5000, len(target_points))
            indices = np.random.choice(len(target_points), n_points, replace=False)
            points = target_points[indices]
            
            # 应用随机变换来模拟不同的姿态
            angle = np.random.uniform(-0.2, 0.2)  # 减小旋转幅度
            rotation = R.from_euler('z', angle).as_matrix()
            translation = np.random.uniform(-0.01, 0.01, 3)  # 减小平移幅度
            
            points = (rotation @ points.T).T + translation
            
            # 添加一些噪声来模拟测量误差
            noise = np.random.normal(0, 0.001, points.shape)
            points += noise
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(self.original_scene_color) # 使用原始场景颜色
            
            return pcd
        else:
            # 备用方案：生成简单的立方体点云
            points = []
            for x in np.linspace(-0.1, 0.1, 15):
                for y in np.linspace(-0.1, 0.1, 15):
                    for z in np.linspace(-0.1, 0.1, 15):
                        if abs(x) > 0.08 or abs(y) > 0.08 or abs(z) > 0.08:
                            points.append([x, y, z])
            
            points = np.array(points)
            if len(points) > 0:
                # 添加一些噪声
                noise = np.random.normal(0, 0.005, points.shape)
                points += noise
                
                # 应用随机变换
                angle = np.random.uniform(-0.3, 0.3)
                rotation = R.from_euler('z', angle).as_matrix()
                translation = np.random.uniform(-0.05, 0.05, 3)
                
                points = (rotation @ points.T).T + translation
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(self.original_scene_color) # 使用原始场景颜色
                
                return pcd
            else:
                # 最后的备用方案：空点云
                pcd = o3d.geometry.PointCloud()
                return pcd
    
    def preprocess_scene_point_cloud(self, scene_pcd):
        """预处理场景点云"""
        processed = copy.deepcopy(scene_pcd)
        
        # 体素下采样
        if self.preprocess_voxel_size > 0:
            processed = processed.voxel_down_sample(self.preprocess_voxel_size)
        
        # 统计离群点移除
        if self.preprocess_sor_k > 0:
            processed, _ = processed.remove_statistical_outlier(
                self.preprocess_sor_k, self.preprocess_sor_std_ratio)
        
        # 最远点采样
        if self.preprocess_fps_n_points > 0 and len(processed.points) > self.preprocess_fps_n_points:
            processed = processed.farthest_point_down_sample(self.preprocess_fps_n_points)
        
        return processed
    
    # ===== 粗配准相关函数 (从part1复制) =====
    def axis_angle_to_matrix(self, axis_angle):
        theta = torch.norm(axis_angle)
        if theta < 1e-6: 
            return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        k = axis_angle / theta
        K = torch.as_tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], 
                           dtype=axis_angle.dtype, device=axis_angle.device)
        R = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype) + \
            torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R
    
    def pytorch_coarse_alignment(self, source_points, target_points):
        """PyTorch粗配准"""
        print("    Running PyTorch-based Coarse Alignment...")
        source_tensor = torch.from_numpy(source_points.astype('float32')).to(self.device)
        target_tensor = torch.from_numpy(target_points.astype('float32')).to(self.device)
        
        rot_vec = nn.Parameter(torch.zeros(3, device=self.device))
        translation = nn.Parameter(torch.zeros(3, device=self.device))
        optimizer = optim.Adam([rot_vec, translation], lr=self.coarse_lr)
        
        for i in range(self.coarse_iterations):
            optimizer.zero_grad()
            R = self.axis_angle_to_matrix(rot_vec)
            transformed_source = (R @ source_tensor.T).T + translation
            dists = torch.cdist(transformed_source, target_tensor)
            loss = torch.min(dists, dim=1)[0].mean()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % self.coarse_print_every == 0 or i == 0:
                print(f"      Coarse Align Iter {i+1}/{self.coarse_iterations}, Loss: {loss.item():.6f}")
        
        final_R = self.axis_angle_to_matrix(rot_vec).detach().cpu().numpy()
        final_t = translation.detach().cpu().numpy()
        
        transform = np.identity(4)
        transform[:3, :3] = final_R
        transform[:3, 3] = final_t
        print("    Coarse alignment on centered clouds finished.")
        return transform
    
    # ===== ICP精配准相关函数 (从part2复制和简化) =====
    def find_nearest_neighbors_torch(self, source_points, target_points):
        dist_sq = torch.cdist(source_points, target_points).pow(2)
        min_dist_sq, indices = torch.min(dist_sq, dim=1)
        return indices, min_dist_sq
    
    def estimate_point_to_point_svd_torch(self, P, Q):
        centroid_P = torch.mean(P, dim=0, keepdim=True)
        centroid_Q = torch.mean(Q, dim=0, keepdim=True)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = P_centered.T @ Q_centered
        
        try:
            U, S, Vt = torch.linalg.svd(H)
        except torch.linalg.LinAlgError:
            T = torch.eye(4, device=P.device, dtype=P.dtype)
            return T
        
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt_copy = Vt.clone()
            Vt_copy[2, :] *= -1
            R = Vt_copy.T @ U.T
        
        t = centroid_Q.T - R @ centroid_P.T
        
        T = torch.eye(4, device=P.device, dtype=P.dtype)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        return T
    
    def pytorch_icp_registration(self, source_points_tensor, target_points_tensor, initial_transform_guess):
        """PyTorch ICP配准"""
        current_transform = initial_transform_guess.clone().to(device=self.device, dtype=source_points_tensor.dtype)
        source_homo = torch.cat([source_points_tensor, 
                                torch.ones(source_points_tensor.shape[0], 1, 
                                         device=self.device, dtype=source_points_tensor.dtype)], dim=1).T
        
        prev_rmse = float('inf')
        fitness = 0.0  # 初始化fitness变量
        current_rmse = float('inf')  # 初始化current_rmse变量
        num_correspondences = 0  # 初始化对应点数量
        
        for i in range(self.icp_max_iter):
            transformed_source_points = (current_transform @ source_homo)[:3, :].T
            
            # 找对应点
            corr_indices_target, dist_sq = self.find_nearest_neighbors_torch(
                transformed_source_points, target_points_tensor)
            
            # 过滤对应点
            valid_mask = dist_sq < (self.icp_threshold ** 2)
            num_correspondences = valid_mask.sum().item()
            
            if num_correspondences < self.icp_min_points:
                print(f"ICP迭代 {i + 1}: 对应点太少 ({num_correspondences}), 停止")
                break
            
            P_orig_corr = source_points_tensor[valid_mask]
            Q_corr_target = target_points_tensor[corr_indices_target[valid_mask]]
            
            current_rmse = torch.sqrt(torch.mean(dist_sq[valid_mask])).item()
            fitness = num_correspondences / source_points_tensor.shape[0]
            
            # 估计新变换
            new_total_transform = self.estimate_point_to_point_svd_torch(P_orig_corr, Q_corr_target)
            transform_update_matrix = new_total_transform @ torch.linalg.inv(current_transform)
            current_transform = new_total_transform
            
            # 检查收敛
            delta_transform_norm = torch.norm(
                transform_update_matrix - torch.eye(4, device=self.device, dtype=source_points_tensor.dtype)).item()
            rmse_diff = abs(prev_rmse - current_rmse)
            
            if i % 100 == 0:
                print(f"ICP迭代 {i + 1}/{self.icp_max_iter}: RMSE: {current_rmse:.6f}, Fitness: {fitness:.6f}")
            
            if i > 10 and (rmse_diff < self.icp_relative_rmse or delta_transform_norm < self.icp_transform_change_threshold):
                print(f"ICP在第{i + 1}次迭代收敛")
                break
            
            prev_rmse = current_rmse
        
        return {
            "transformation": current_transform.cpu().numpy(),
            "fitness": fitness,
            "inlier_rmse": current_rmse,
            "correspondence_set_size": num_correspondences
        }
    
    def estimate_pose(self, scene_pcd):
        """估计姿态"""
        try:
            # 预处理场景点云
            scene_preprocessed = self.preprocess_scene_point_cloud(scene_pcd)
            
            if len(scene_preprocessed.points) < self.icp_min_points:
                print("警告: 预处理后的点云太少，跳过姿态估计")
                return None, None, None
            
            print(f"开始姿态估计，场景点数: {len(scene_preprocessed.points)}")
            
            # 1. 中心化点云
            scene_centroid = scene_preprocessed.get_center()
            scene_centered = copy.deepcopy(scene_preprocessed).translate(-scene_centroid)
            
            target_centroid = self.target_pcd_for_matching.get_center()
            target_centered = copy.deepcopy(self.target_pcd_for_matching).translate(-target_centroid)
            
            # 2. 粗配准
            print("开始粗配准...")
            source_pcd_for_align = scene_centered
            target_pcd_for_align = target_centered
            
            if len(source_pcd_for_align.points) > self.coarse_align_scene_points:
                source_pcd_for_align = source_pcd_for_align.farthest_point_down_sample(self.coarse_align_scene_points)
            
            if len(target_pcd_for_align.points) > self.coarse_align_model_points:
                target_pcd_for_align = target_pcd_for_align.farthest_point_down_sample(self.coarse_align_model_points)
            
            coarse_transform_centered = self.pytorch_coarse_alignment(
                np.asarray(source_pcd_for_align.points),
                np.asarray(target_pcd_for_align.points)
            )
            
            # 3. ICP精配准
            print("开始ICP精配准...")
            source_points_tensor = torch.from_numpy(
                np.asarray(scene_centered.points).astype(np.float32)).float().to(self.device)
            target_points_tensor = torch.from_numpy(
                np.asarray(target_centered.points).astype(np.float32)).float().to(self.device)
            initial_transform_tensor = torch.from_numpy(coarse_transform_centered).to(
                device=self.device, dtype=source_points_tensor.dtype)
            
            icp_result = self.pytorch_icp_registration(
                source_points_tensor, target_points_tensor, initial_transform_tensor)
            
            # 4. 计算最终变换
            T_centered = icp_result["transformation"]
            
            # 构建最终变换矩阵
            T_to_scene_centroid = np.identity(4)
            T_to_scene_centroid[:3, 3] = scene_centroid
            
            T_from_target_centroid = np.identity(4)
            T_from_target_centroid[:3, 3] = -target_centroid
            
            final_transform = T_to_scene_centroid @ np.linalg.inv(T_centered) @ T_from_target_centroid
            
            # 计算粗配准的最终变换（用于可视化）
            coarse_final_transform = T_to_scene_centroid @ np.linalg.inv(coarse_transform_centered) @ T_from_target_centroid
            
            print(f"姿态估计完成 - Fitness: {icp_result['fitness']:.4f}, RMSE: {icp_result['inlier_rmse']:.6f}")
            
            # 返回结果信息
            matching_info = {
                'fitness': icp_result['fitness'],
                'inlier_rmse': icp_result['inlier_rmse'],
                'correspondence_set_size': icp_result['correspondence_set_size'],
                'preprocessed_points': len(scene_preprocessed.points)
            }
            
            return final_transform, coarse_final_transform, matching_info
            
        except Exception as e:
            print(f"姿态估计失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def run(self):
        """主运行循环"""
        print("=== 实时点云姿态估计与STL模型可视化 ===")
        print("关闭可视化窗口或按任意键退出程序")
        print(f"程序将每{self.pose_interval}秒进行一次姿态估计")
        
        # 显示配置信息
        print("\n=== 显示配置 ===")
        print(f"原始场景点云: {'✓' if self.show_original_scene else '✗'}")
        print(f"预处理场景点云: {'✓' if self.show_preprocessed_scene else '✗'}")
        print(f"目标模型: {'✓' if self.show_target_model else '✗'}")
        print(f"匹配后模型: {'✓' if self.show_matched_model else '✗'}")
        print(f"模型采样点云: {'✓' if self.show_model_pointcloud else '✗'}")
        print(f"粗配准结果: {'✓' if self.show_coarse_alignment else '✗'}")
        print(f"对应点连线: {'✓' if self.show_correspondences else '✗'}")
        print(f"边界框: {'✓' if self.show_bounding_boxes else '✗'}")
        print(f"中心点: {'✓' if self.show_center_points else '✗'}")
        print(f"数据保存: {'✓' if self.enable_saving else '✗'}")
        print(f"2D图像显示: {'✓' if self.enable_2d_display else '✗'}")
        print("=" * 50)
        
        # 连接相机
        camera_connected = self.connect_camera()
        if not camera_connected:
            print("无法连接相机，将使用虚拟数据进行演示")
        
        print("开始实时姿态估计...")
        
        frame_count = 0
        last_pose_time = 0
        
        try:
            while True:
                current_time = time.time()
                
                # 捕获数据
                if camera_connected:
                    scene_pcd, rgb_image, depth_image = self.capture_point_cloud_from_camera()
                else:
                    # 使用虚拟数据
                    scene_pcd = self.generate_virtual_point_cloud()
                    rgb_image, depth_image = None, None
                    
                if scene_pcd is None or not scene_pcd.has_points():
                    if camera_connected:
                        print("无法获取有效的点云数据，等待相机准备...")
                        time.sleep(1.0)
                        continue
                    else:
                        print("虚拟数据生成失败，程序退出")
                        break
                
                # 存储当前帧数据
                self.current_frame_data = {
                    'scene_pcd': scene_pcd,
                    'rgb_image': rgb_image,
                    'depth_image': depth_image,
                    'preprocessed_pcd': None,
                    'estimated_pose': None,
                    'coarse_transform': None,
                    'matching_info': None
                }
                
                estimated_pose, coarse_transform, matching_info = None, None, None
                
                # 定期进行姿态估计
                if current_time - last_pose_time > self.pose_interval:
                    print(f"\n--- 第 {frame_count + 1} 帧姿态估计 ---")
                    estimated_pose, coarse_transform, matching_info = self.estimate_pose(scene_pcd)
                    last_pose_time = current_time
                    
                    # 更新帧数据
                    self.current_frame_data.update({
                        'estimated_pose': estimated_pose,
                        'coarse_transform': coarse_transform,
                        'matching_info': matching_info,
                        'preprocessed_pcd': self.preprocess_scene_point_cloud(scene_pcd)
                    })
                
                # 更新可视化
                self.update_visualization(
                    scene_pcd=scene_pcd,
                    preprocessed_pcd=self.current_frame_data['preprocessed_pcd'],
                    estimated_pose=estimated_pose,
                    coarse_transform=coarse_transform,
                    matching_info=matching_info
                )
                
                # 保存帧数据
                if self.enable_saving:
                    self.save_frame_data(
                        frame_count, 
                        scene_pcd, 
                        self.current_frame_data['preprocessed_pcd'],
                        rgb_image, 
                        depth_image, 
                        estimated_pose, 
                        matching_info
                    )
                
                # 显示2D图像
                self.display_2d_images(rgb_image, depth_image)
                
                # 检查退出条件
                if self.should_exit or not self.vis.poll_events():
                    print("程序退出 (窗口关闭或用户输入)")
                    break
                
                frame_count += 1
                time.sleep(self.frame_delay)
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            if self.vis:
                self.vis.destroy_window()
            if MECHEYE_AVAILABLE and self.camera:
                self.camera.disconnect()
                print("相机已断开连接")
            if self.enable_2d_display and hasattr(self, 'cv2'):
                self.cv2.destroyAllWindows()
                print("2D图像窗口已关闭")

def main():
    """主函数"""
    try:
        # 使用配置文件创建应用
        config_file = "realtime_pose_config.yaml"
        app = RealtimePoseEstimation(config_file)
        
        # 检查STL文件是否存在
        if not os.path.exists(app.model_file):
            print(f"错误: 找不到STL文件 {app.model_file}")
            sys.exit(1)
        
        app.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 