#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mech-Eye相机全面测试程序
包括重复精度、帧率、稳定性等测试
"""

import time
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class NumpyEncoder(json.JSONEncoder):
    """ 自定义编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    success: bool
    data: Dict
    error_message: Optional[str] = None
    execution_time: float = 0.0


class MechEyeCameraTester:
    """Mech-Eye相机测试类"""
    def __init__(self):
        self.camera = Camera()
        self.test_results: List[TestResult] = []
        self.logger = logging.getLogger("MechEyeCameraTester")
        self.logger.setLevel(logging.INFO)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_params = {
            'frame_rate_test_duration': 500,
            'repeatability_test_count': 500,
            'stability_test_duration': 1000,
            'noise_test_count': 500,
        }
    
    def setup_camera(self) -> bool:
        """设置相机连接"""
        if find_and_connect(self.camera):
            self.logger.info("相机连接成功")
            return True
        else:
            self.logger.error("相机连接失败")
            return False
    
    def run_test(self, test_func, test_name: str, **kwargs) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        result = TestResult(test_name=test_name, success=False, data={})
        self.logger.info(f"开始测试: {test_name}")
        data = test_func(**kwargs)
        result.data = data
        result.success = True
        self.logger.info(f"测试完成: {test_name}")
        result.execution_time = time.time() - start_time
        self.test_results.append(result)
        return result
    
    def test_basic_functionality(self) -> Dict:
        """基础功能测试，完全模仿capture_2d_image.py和capture_point_cloud.py风格"""
        data = {}
        from mecheye.area_scan_3d_camera import CameraInfo
        info = CameraInfo()
        self.camera.get_camera_info(info)
        data['camera_info'] = {
            'model': getattr(info, 'model', None),
            'ip': getattr(info, 'ip_address', None),
            'sn': getattr(info, 'serial_number', None)
        }
        # 采集2D图像
        frame_2d = Frame2D()
        self.camera.capture_2d(frame_2d)
        image2d = None
        file_name_2d = self.results_dir / "2DImage.png"
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()
        if image2d is not None:
            cv2.imwrite(str(file_name_2d), image2d.data())
            print(f"Capture and save the 2D image: {file_name_2d}")
            data['image_shape'] = (image2d.height(), image2d.width()) if len(image2d.data().shape)==2 else (image2d.height(), image2d.width(), 3)
        # 采集点云
        frame_all_2d_3d = Frame2DAnd3D()
        self.camera.capture_2d_and_3d(frame_all_2d_3d)
        # 保存无纹理点云
        point_cloud_file = self.results_dir / "PointCloud.ply"
        frame_all_2d_3d.frame_3d().save_untextured_point_cloud(FileFormat_PLY, str(point_cloud_file))
        print(f"Capture and save the untextured point cloud: {point_cloud_file}.")
        # 保存有纹理点云
        textured_point_cloud_file = self.results_dir / "TexturedPointCloud.ply"
        frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY, str(textured_point_cloud_file))
        print(f"Capture and save the textured point cloud: {textured_point_cloud_file}.")
        data['pointcloud_file'] = str(point_cloud_file)
        data['textured_pointcloud_file'] = str(textured_point_cloud_file)
        return data
    
    def test_frame_rate(self) -> Dict:
        """帧率测试，宽松风格"""
        duration = self.test_params['frame_rate_test_duration']
        frame_times = []
        success_count = 0
        total_count = 0
        start_time = time.time()
        end_time = start_time + duration
        self.logger.info(f"开始帧率测试，持续时间: {duration}秒")
        while time.time() < end_time:
            frame_start = time.time()
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)
            success_count += 1
            time.sleep(0.5)
            total_count += 1
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        std_frame_time = np.std(frame_times) if frame_times else 0
        min_frame_time = np.min(frame_times) if frame_times else 0
        max_frame_time = np.max(frame_times) if frame_times else 0
        fps = success_count / duration if duration > 0 else 0
        fps_filtered = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        print(f"Frame rate test: {success_count} frames in {duration} seconds, FPS: {fps:.2f}")
        return {
            'duration': duration,
            'total_attempts': total_count,
            'successful_captures': success_count,
            'fps': fps,
            'fps_filtered': fps_filtered,
            'avg_frame_time': avg_frame_time,
            'std_frame_time': std_frame_time,
            'min_frame_time': min_frame_time,
            'max_frame_time': max_frame_time,
            'frame_times': frame_times
        }
    
    def test_repeatability(self) -> Dict:
        """重复精度测试，宽松风格"""
        count = self.test_params['repeatability_test_count']
        point_clouds = []
        center_points = []
        selected_points = []
        success_count = 0
        self.logger.info(f"开始重复精度测试，测试次数: {count}")
        for i in range(count):
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_3d = frame_2d_and_3d.frame_3d()
            point_cloud = frame_3d.get_untextured_point_cloud()
            if point_cloud:
                pc_np = point_cloud.data()
                if pc_np.ndim == 3:
                    pc_np = pc_np.reshape(-1, pc_np.shape[-1])
                valid_points = pc_np[~np.isnan(pc_np).any(axis=1)]
                valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
                if len(valid_points) > 0:
                    center = np.mean(valid_points, axis=0)
                    center_points.append(center)
                    point_clouds.append(pc_np)
                    if len(valid_points) >= 20:
                        indices = np.linspace(0, len(valid_points)-1, 20, dtype=int)
                        selected_pts = valid_points[indices]
                        selected_points.append(selected_pts)
                    else:
                        selected_points.append(valid_points)
                    success_count += 1
                else:
                    self.logger.warning(f"第{i+1}次点云无有效点")
            else:
                self.logger.warning(f"第{i+1}次点云为空")
            time.sleep(0.5)
        if len(center_points) < 2:
            return {'error': '有效数据不足，无法计算重复精度'}
        center_points = np.array(center_points)
        selected_points = np.array(selected_points)
        mean_center = np.mean(center_points, axis=0)
        std_center = np.std(center_points, axis=0)
        max_deviation = np.max(np.linalg.norm(center_points - mean_center, axis=1))
        print(f"Repeatability test: max deviation = {max_deviation:.3f} mm")
        return {
            'test_count': count,
            'success_count': success_count,
            'max_deviation': max_deviation,
            'mean_center': mean_center.tolist(),
            'std_center': std_center.tolist(),
            'center_points': center_points.tolist()
        }
    
    def test_stability(self) -> Dict:
        """稳定性测试，宽松风格"""
        duration = self.test_params['stability_test_duration']
        interval = 1.0
        measurements = []
        start_time = time.time()
        self.logger.info(f"开始稳定性测试，持续时间: {duration}秒")
        while time.time() - start_time < duration:
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_3d = frame_2d_and_3d.frame_3d()
            point_cloud = frame_3d.get_untextured_point_cloud()
            depth_map = frame_3d.get_depth_map()
            measurement = {
                'timestamp': time.time() - start_time,
                'has_pointcloud': point_cloud is not None,
                'has_depth': depth_map is not None
            }
            if point_cloud:
                pc_np = point_cloud.data()
                if pc_np.ndim == 3:
                    pc_np = pc_np.reshape(-1, pc_np.shape[-1])
                valid_points = pc_np[~np.isnan(pc_np).any(axis=1)]
                valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
                measurement['valid_points'] = len(valid_points)
                if len(valid_points) > 0:
                    measurement['mean_distance'] = float(np.mean(np.linalg.norm(valid_points, axis=1)))
            measurements.append(measurement)
            time.sleep(0.5)
        print(f"Stability test: {len(measurements)} measurements in {duration} seconds")
        return {
            'duration': duration,
            'total_measurements': len(measurements),
            'measurements': measurements[:100]
        }
    
    def test_noise(self) -> Dict:
        """噪声测试，宽松风格"""
        count = self.test_params['noise_test_count']
        depth_maps = []
        success_count = 0
        self.logger.info(f"开始噪声测试，测试次数: {count}")
        for i in range(count):
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_3d = frame_2d_and_3d.frame_3d()
            depth_map = frame_3d.get_depth_map()
            if depth_map:
                depth_np = depth_map.data()
                depth_maps.append(depth_np)
                success_count += 1
            else:
                self.logger.warning(f"第{i+1}次深度图为空")
            time.sleep(0.5)
        if len(depth_maps) < 2:
            return {'error': '有效数据不足，无法分析噪声'}
        depth_maps = np.array(depth_maps)
        mean_depth = np.mean(depth_maps, axis=0)
        std_depth = np.std(depth_maps, axis=0)
        valid_mask = ~np.isnan(mean_depth) & ~np.isinf(mean_depth)
        noise_mean = np.mean(std_depth[valid_mask]) if np.any(valid_mask) else 0
        print(f"Noise test: mean noise = {noise_mean:.4f}")
        return {
            'test_count': count,
            'success_count': success_count,
            'noise_mean': noise_mean
        }
    
    def test_accuracy(self) -> Dict:
        """精度测试，宽松风格"""
        test_count = 3
        measurements = []
        selected_points_series = []
        self.logger.info("开始精度测试")
        for i in range(test_count):
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_3d = frame_2d_and_3d.frame_3d()
            point_cloud = frame_3d.get_untextured_point_cloud()
            if point_cloud:
                pc_np = point_cloud.data()
                if pc_np.ndim == 3:
                    pc_np = pc_np.reshape(-1, pc_np.shape[-1])
                valid_points = pc_np[~np.isnan(pc_np).any(axis=1)]
                valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
                if len(valid_points) > 0:
                    center = np.mean(valid_points, axis=0)
                    measurements.append(center)
                    if len(valid_points) >= 20:
                        indices = np.linspace(0, len(valid_points)-1, 20, dtype=int)
                        selected_pts = valid_points[indices]
                        selected_points_series.append(selected_pts)
                    else:
                        selected_points_series.append(valid_points)
            else:
                self.logger.warning(f"第{i+1}次点云为空")
            time.sleep(0.5)
        if len(measurements) < 2:
            return {'error': '有效测量数据不足'}
        measurements = np.array(measurements)
        mean_measurement = np.mean(measurements, axis=0)
        std_measurement = np.std(measurements, axis=0)
        print(f"Accuracy test: mean = {mean_measurement.tolist()}, std = {std_measurement.tolist()}")
        return {
            'test_count': test_count,
            'measurement_count': len(measurements),
            'mean_measurement': mean_measurement.tolist(),
            'std_measurement': std_measurement.tolist()
        }
    
    def run_all_tests(self) -> List[TestResult]:
        """运行所有测试"""
        if not self.setup_camera():
            self.logger.error("相机设置失败，无法进行测试")
            return []
        # 基础功能测试
        self.run_test(self.test_basic_functionality, "基础功能测试")
        # 帧率测试
        self.run_test(self.test_frame_rate, "帧率测试")
        # 重复精度测试
        self.run_test(self.test_repeatability, "重复精度测试")
        # 稳定性测试
        self.run_test(self.test_stability, "稳定性测试")
        # 噪声测试
        self.run_test(self.test_noise, "噪声测试")
        # 精度测试
        self.run_test(self.test_accuracy, "精度测试")
        if self.camera:
            self.camera.disconnect()
        return self.test_results
    
    def save_results(self, filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"test_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # 准备保存的数据
        save_data = {
            'test_timestamp': time.time(),
            'camera_ip': None,
            'test_params': self.test_params,
            'results': []
        }
        
        from mecheye.area_scan_3d_camera import CameraInfo
        info = CameraInfo()
        self.camera.get_camera_info(info)
        save_data['camera_ip'] = getattr(info, 'ip_address', None)
        
        for result in self.test_results:
            result_data = {
                'test_name': result.test_name,
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'data': result.data
            }
            save_data['results'].append(result_data)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.logger.info(f"测试结果已保存: {filepath}")
        return filepath
    
    def generate_report(self, output_file: str = None):
        """生成测试报告"""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"test_report_{timestamp}.md"
        
        filepath = self.results_dir / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# MechEye相机测试报告\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            from mecheye.area_scan_3d_camera import CameraInfo
            info = CameraInfo()
            self.camera.get_camera_info(info)
            ip = getattr(info, 'ip_address', None)
            f.write(f"相机IP: {ip}\n\n")
            
            # 测试参数
            f.write("## 测试参数\n\n")
            for key, value in self.test_params.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            # 测试结果摘要
            f.write("## 测试结果摘要\n\n")
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results if r.success)
            f.write(f"- 总测试数: {total_tests}\n")
            f.write(f"- 成功测试数: {successful_tests}\n")
            f.write(f"- 成功率: {successful_tests/total_tests*100:.1f}%\n\n")
            
            # 详细结果
            f.write("## 详细测试结果\n\n")
            for result in self.test_results:
                f.write(f"### {result.test_name}\n\n")
                f.write(f"- 状态: {'成功' if result.success else '失败'}\n")
                f.write(f"- 执行时间: {result.execution_time:.2f}秒\n")
                
                if result.error_message:
                    f.write(f"- 错误信息: {result.error_message}\n")
                
                if result.success and result.data:
                    f.write("- 测试数据:\n")
                    f.write("```json\n")
                    f.write(json.dumps(result.data, indent=2, ensure_ascii=False, cls=NumpyEncoder))
                    f.write("\n```\n")
                
                f.write("\n")
        
        self.logger.info(f"测试报告已生成: {filepath}")
        return filepath


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试器
    tester = MechEyeCameraTester()
    
    # 运行所有测试
    print("开始MechEye相机全面测试...")
    results = tester.run_all_tests()
    
    # 保存结果
    results_file = tester.save_results()
    
    # 生成报告
    report_file = tester.generate_report()
    
    # 打印摘要
    print("\n" + "="*50)
    print("测试完成！")
    print(f"结果文件: {results_file}")
    print(f"报告文件: {report_file}")
    
    successful_tests = sum(1 for r in results if r.success)
    print(f"成功测试: {successful_tests}/{len(results)}")
    
    # 显示关键结果
    for result in results:
        if result.success and result.data:
            if result.test_name == "帧率测试":
                fps = result.data.get('fps', 0)
                print(f"帧率: {fps:.2f} FPS")
            elif result.test_name == "重复精度测试":
                max_dev = result.data.get('max_deviation', 0)
                print(f"重复精度: {max_dev:.3f} mm")


if __name__ == "__main__":
    main() 