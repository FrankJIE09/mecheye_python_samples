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
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas未安装，Excel保存功能将不可用")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: tqdm未安装，进度条显示将不可用")

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class NumpyEncoder(json.JSONEncoder):
    """ 自定义编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if hasattr(obj, 'dtype') and obj.dtype.kind in 'iu':  # integer types
            return int(obj)
        elif hasattr(obj, 'dtype') and obj.dtype.kind == 'f':  # float types
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
            'frame_rate_test_duration': 0,
            'repeatability_test_count': 100,
            'stability_test_duration': 0,
            'noise_test_count': 0,
        }
    
    def setup_camera(self) -> bool:
        """设置相机连接"""
        if find_and_connect(self.camera):
            self.logger.info("相机连接成功")
            return True
        else:
            self.logger.error("相机连接失败")
            return False
    
    def capture_single_frame(self):
        """捕获单帧深度图像"""
        try:
            frame_2d_and_3d = Frame2DAnd3D()
            self.camera.capture_2d_and_3d(frame_2d_and_3d)
            frame_3d = frame_2d_and_3d.frame_3d()
            depth_map = frame_3d.get_depth_map()
            if depth_map:
                return depth_map.data()
            else:
                return None
        except Exception as e:
            self.logger.error(f"捕获深度图像失败: {e}")
            return None
    
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
        """重复精度测试 - 多点测试
        注意：MechEye相机的深度数据本身就是毫米(mm)单位，无需比例因子转换
        """
        count = self.test_params['repeatability_test_count']
        target_points = 25  # 目标测试点数
        depth_maps = []
        all_test_points = []  # 存储所有测试点的数据
        success_count = 0

        self.logger.info(f"开始重复精度测试，测试次数: {count}，目标测试点数: {target_points}")

        # 首先获取一帧来确定有效区域和采样点
        initial_depth = self.capture_single_frame()
        if initial_depth is None:
            return {'error': '无法获取初始深度图像'}

        # 计算有效区域
        valid_mask = (initial_depth > 0) & (initial_depth < 65535)
        if not np.any(valid_mask):
            return {'error': '未找到有效深度区域'}

        valid_coords = np.where(valid_mask)
        
        # 在有效区域内均匀采样点
        if len(valid_coords[0]) >= target_points:
            # 随机选择目标数量的点
            indices = np.random.choice(len(valid_coords[0]), target_points, replace=False)
            test_y_coords = valid_coords[0][indices]
            test_x_coords = valid_coords[1][indices]
        else:
            # 如果有效点不够，使用所有有效点
            test_y_coords = valid_coords[0]
            test_x_coords = valid_coords[1]
            target_points = len(test_y_coords)
            self.logger.warning(f"有效点数量不足，使用所有 {target_points} 个有效点")

        self.logger.info(f"选择了 {target_points} 个测试点")

        # 为每个测试点创建数据存储
        point_data = {i: [] for i in range(target_points)}

        # 使用tqdm显示进度（如果可用）
        if TQDM_AVAILABLE:
            progress_bar = tqdm(total=count, desc="重复精度测试", unit="次", ncols=80)
        else:
            progress_bar = None
            
        for i in range(count):
            # 在每次拍照前等待1秒，确保获取新图像
            time.sleep(1)
            
            # 捕获深度图像
            depth_data = self.capture_single_frame()

            if depth_data is not None:
                # 记录每个测试点的深度值
                for j in range(target_points):
                    y, x = test_y_coords[j], test_x_coords[j]
                    if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
                        depth_value = depth_data[y, x]
                        if depth_value > 0 and depth_value < 65535:
                            # 深度数据本身就是毫米单位，不需要比例因子转换
                            depth_mm = depth_value  # 直接使用原始值，单位已经是毫米
                            point_data[j].append(depth_mm)

                depth_maps.append(depth_data)
                success_count += 1

                self.logger.debug(f"重复精度测试进度: {i + 1}/{count}")
            else:
                self.logger.warning(f"第{i + 1}次拍摄失败")

            # 更新进度条
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    '成功': success_count,
                    '成功率': f"{success_count/(i+1)*100:.1f}%"
                })
            else:
                print(f"进度: {i+1}/{count}, 成功: {success_count}")
        
        if progress_bar:
            progress_bar.close()

        # 分析每个测试点的重复精度
        point_analysis = []
        valid_points = 0

        for i in range(target_points):
            if len(point_data[i]) >= 2:  # 至少需要2个有效测量值
                # 现在point_data[i]只包含深度值
                try:
                    depths = np.array(point_data[i], dtype=np.float64)
                    
                    # 计算该点的统计信息
                    mean_depth = np.mean(depths)
                    std_depth = np.std(depths)
                    max_depth_deviation = np.max(np.abs(depths - mean_depth))
                    depth_range = np.max(depths) - np.min(depths)
                    
                    point_analysis.append({
                        'point_id': i,
                        'coordinates': [test_x_coords[i], test_y_coords[i]],
                        'measurement_count': len(depths),
                        'mean_depth': mean_depth,
                        'std_depth': std_depth,
                        'max_depth_deviation': max_depth_deviation,
                        'depth_range': depth_range
                    })
                    valid_points += 1
                except Exception as e:
                    self.logger.warning(f"处理测试点 {i} 时出错: {e}")
                    continue

        if valid_points == 0:
            return {'error': '没有足够的有效测试点数据'}

        # 计算整体统计
        all_depth_stds = [p['std_depth'] for p in point_analysis]
        all_depth_deviations = [p['max_depth_deviation'] for p in point_analysis]
        all_depth_ranges = [p['depth_range'] for p in point_analysis]
        
        overall_stats = {
            'mean_depth_std': np.mean(all_depth_stds),
            'std_depth_std': np.std(all_depth_stds),
            'max_depth_std': np.max(all_depth_stds),
            'mean_depth_deviation': np.mean(all_depth_deviations),
            'std_depth_deviation': np.std(all_depth_deviations),
            'max_depth_deviation': np.max(all_depth_deviations),
            'mean_depth_range': np.mean(all_depth_ranges),
            'std_depth_range': np.std(all_depth_ranges)
        }

        # 保存数据到Excel
        self.save_repeatability_excel(point_data, point_analysis, overall_stats, count)

        return {
            'test_count': count,
            'success_count': success_count,
            'success_rate': success_count / count,
            'target_points': target_points,
            'valid_points': valid_points,
            'overall_stats': overall_stats,
            'point_analysis': point_analysis,
            'test_point_coordinates': [[test_x_coords[i], test_y_coords[i]] for i in range(target_points)],
            'raw_depth_data': point_data  # 保存每个点的原始深度数据
        }
    
    def save_repeatability_excel(self, point_data: Dict, point_analysis: List, overall_stats: Dict, test_count: int):
        """保存重复精度测试数据到Excel文件"""
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas未安装，无法保存Excel文件")
            return None
            
        timestamp = int(time.time())
        excel_filename = f"repeatability_test_{timestamp}.xlsx"
        excel_filepath = self.results_dir / excel_filename
        
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
                
                # 1. 原始数据表
                raw_data_rows = []
                for point_id, depths in point_data.items():
                    for measurement_idx, depth in enumerate(depths):
                        raw_data_rows.append({
                            'Point_ID': point_id,
                            'Measurement_Index': measurement_idx + 1,
                            'Depth_mm': depth
                        })
                
                if raw_data_rows:
                    raw_df = pd.DataFrame(raw_data_rows)
                    raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # 2. 点分析表
                if point_analysis:
                    analysis_rows = []
                    for point in point_analysis:
                        analysis_rows.append({
                            'Point_ID': point['point_id'],
                            'X_Coordinate': point['coordinates'][0],
                            'Y_Coordinate': point['coordinates'][1],
                            'Measurement_Count': point['measurement_count'],
                            'Mean_Depth_mm': point['mean_depth'],
                            'Std_Depth_mm': point['std_depth'],
                            'Max_Deviation_mm': point['max_depth_deviation'],
                            'Depth_Range_mm': point['depth_range']
                        })
                    
                    analysis_df = pd.DataFrame(analysis_rows)
                    analysis_df.to_excel(writer, sheet_name='Point_Analysis', index=False)
                
                # 3. 整体统计表
                stats_rows = []
                for key, value in overall_stats.items():
                    stats_rows.append({
                        'Statistic': key,
                        'Value': value
                    })
                
                stats_df = pd.DataFrame(stats_rows)
                stats_df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
                
                # 4. 测试信息表
                info_rows = [
                    {'Parameter': 'Test_Count', 'Value': test_count},
                    {'Parameter': 'Target_Points', 'Value': len(point_data)},
                    {'Parameter': 'Valid_Points', 'Value': len(point_analysis)},
                    {'Parameter': 'Test_Date', 'Value': time.strftime('%Y-%m-%d %H:%M:%S')}
                ]
                
                info_df = pd.DataFrame(info_rows)
                info_df.to_excel(writer, sheet_name='Test_Info', index=False)
            
            self.logger.info(f"重复精度测试数据已保存到Excel: {excel_filepath}")
            return excel_filepath
        except Exception as e:
            self.logger.error(f"保存Excel文件失败: {e}")
            return None
    
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