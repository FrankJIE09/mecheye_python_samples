#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时RGB图像显示与点云拍摄工具
功能：
- 实时显示RGB图像
- 按's'键拍摄点云图并保存到指定文件夹
- 按'q'键退出程序
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import sys

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d

class RealtimeRGBPointCloudCapture:
    def __init__(self):
        self.camera = Camera()
        self.output_folder = "captured_pointclouds"
        self.capture_count = 0
        
        # 创建输出文件夹
        self._create_output_folder()
        
    def _create_output_folder(self):
        """创建保存点云图的文件夹"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"创建输出文件夹: {self.output_folder}")
    
    def connect_camera(self):
        """连接相机"""
        try:
            # 使用find_and_connect函数连接相机
            if find_and_connect(self.camera):
                print("相机连接成功")
                return True
            else:
                print("错误：未找到可用的相机")
                return False
        except Exception as e:
            print(f"连接相机时出错: {e}")
            return False
    
    def capture_rgb_image(self):
        """捕获RGB图像"""
        try:
            if self.camera is None:
                return None
            
            # 捕获2D图像
            frame_2d = Frame2D()
            show_error(self.camera.capture_2d(frame_2d))
            
            if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
                image2d = frame_2d.get_gray_scale_image()
                # 转换为彩色图像用于显示
                rgb_image = cv2.cvtColor(image2d.data(), cv2.COLOR_GRAY2BGR)
            elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
                image2d = frame_2d.get_color_image()
                # 转换为BGR格式（OpenCV默认格式）
                rgb_image = cv2.cvtColor(image2d.data(), cv2.COLOR_RGB2BGR)
            else:
                print("警告：未知的图像类型")
                return None
                
            return rgb_image
        except Exception as e:
            print(f"捕获RGB图像时出错: {e}")
            return None
    
    def capture_point_cloud(self):
        """捕获点云数据"""
        try:
            if self.camera is None:
                return None
            
            # 确认3D捕获
            if not confirm_capture_3d():
                print("用户取消3D捕获")
                return None
            
            # 捕获2D和3D数据
            frame_2d_and_3d = Frame2DAnd3D()
            show_error(self.camera.capture_2d_and_3d(frame_2d_and_3d))
            
            # 获取3D帧数据
            frame_3d = frame_2d_and_3d.frame_3d()
            return frame_3d
        except Exception as e:
            print(f"捕获点云时出错: {e}")
            return None
    
    def save_point_cloud(self, frame_3d):
        """保存点云数据到文件"""
        if frame_3d is None:
            return False
        
        try:
            # 生成文件名（时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_{timestamp}.ply"
            filepath = os.path.join(self.output_folder, filename)
            
            # 保存为PLY格式（不带纹理）
            success_message = f"点云已保存: {filepath}"
            show_error(frame_3d.save_untextured_point_cloud(FileFormat_PLY, filepath), success_message)
            
            self.capture_count += 1
            return True
        except Exception as e:
            print(f"保存点云时出错: {e}")
            return False
    
    def run(self):
        """主运行循环"""
        print("=== 实时RGB图像显示与点云拍摄工具 ===")
        print("按 's' 键拍摄点云图")
        print("按 'q' 键退出程序")
        print("=" * 40)
        
        # 连接相机
        if not self.connect_camera():
            print("无法连接相机，程序退出")
            return
        
        # 创建窗口
        cv2.namedWindow("实时RGB图像", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("实时RGB图像", 800, 600)
        
        print("开始实时显示...")
        
        while True:
            # 捕获RGB图像
            rgb_image = self.capture_rgb_image()
            
            if rgb_image is not None:
                # 显示图像
                cv2.imshow("实时RGB图像", rgb_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("用户按下q键，程序退出")
                break
            elif key == ord('s'):
                print("用户按下s键，开始拍摄点云...")
                
                # 捕获点云
                frame_3d = self.capture_point_cloud()
                if frame_3d is not None:
                    # 保存点云
                    if self.save_point_cloud(frame_3d):
                        print(f"点云拍摄成功！已保存 {self.capture_count} 个点云文件")
                    else:
                        print("点云拍摄失败")
                else:
                    print("无法捕获点云数据")
            
            # 短暂延迟
            time.sleep(0.01)
        
        # 清理资源
        cv2.destroyAllWindows()
        if self.camera is not None:
            self.camera.disconnect()
            print("相机已断开连接")
        
        print(f"程序结束，共拍摄了 {self.capture_count} 个点云文件")

def main():
    """主函数"""
    try:
        app = RealtimeRGBPointCloudCapture()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 