#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel表格重复定位精度计算程序
用于分析从EpicEye相机测试导出的Excel数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExcelPrecisionCalculator:
    """Excel表格重复定位精度计算器"""
    
    def __init__(self, excel_file: str):
        """
        初始化计算器
        
        Args:
            excel_file: Excel文件路径
        """
        self.excel_file = excel_file
        self.data = None
        self.results = {}
        
    def load_data(self) -> bool:
        """加载Excel数据"""
        try:
            self.data = pd.read_excel(self.excel_file)
            print(f"成功加载Excel文件: {self.excel_file}")
            print(f"数据形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"加载Excel文件失败: {e}")
            return False
    
    def analyze_data_structure(self) -> Dict:
        """分析数据结构"""
        if self.data is None:
            return {}
        
        analysis = {
            'total_rows': len(self.data),
            'columns': list(self.data.columns),
            'frame_count': self.data['Measurement_Index'].nunique() if 'Measurement_Index' in self.data.columns else 0,
            'point_count': self.data['Point_ID'].nunique() if 'Point_ID' in self.data.columns else 0,
            'has_coordinates': all(col in self.data.columns for col in ['X', 'Y', 'Depth_mm']),
            'data_types': self.data.dtypes.to_dict()
        }
        
        print("\n=== 数据结构分析 ===")
        print(f"总行数: {analysis['total_rows']}")
        print(f"帧数: {analysis['frame_count']}")
        print(f"每帧点数: {analysis['point_count']}")
        print(f"包含坐标: {analysis['has_coordinates']}")
        
        return analysis
    
    def calculate_center_point_precision(self) -> Dict:
        """计算中心点Z方向重复定位精度"""
        if self.data is None or 'Depth_mm' not in self.data.columns:
            return {'error': '数据格式不正确，需要Z列'}
        
        # 检查是否有像素坐标列，如果有则按像素坐标分组计算中心点
        if '像素坐标X' in self.data.columns and '像素坐标Y' in self.data.columns:
            # 按像素坐标分组计算中心点Z坐标
            center_z_values = []
            frame_numbers = []
            
            # 按帧分组
            for frame_id in self.data['Measurement_Index'].unique():
                frame_data = self.data[self.data['Measurement_Index'] == frame_id]
                center_z = frame_data['Depth_mm'].mean()
                center_z_values.append(center_z)
                frame_numbers.append(frame_id)
        else:
            # 按帧分组计算中心点Z坐标（兼容旧格式）
            center_z_values = []
            frame_numbers = []
            
            for frame_id in self.data['Measurement_Index'].unique():
                frame_data = self.data[self.data['Measurement_Index'] == frame_id]
                center_z = frame_data['Depth_mm'].mean()
                center_z_values.append(center_z)
                frame_numbers.append(frame_id)
        
        center_z_values = np.array(center_z_values)
        
        # 计算Z方向统计指标
        mean_z = np.mean(center_z_values)
        std_z = np.std(center_z_values)
        max_deviation_z = np.max(np.abs(center_z_values - mean_z))
        min_deviation_z = np.min(np.abs(center_z_values - mean_z))
        
        # 计算相对误差
        relative_error_z = max_deviation_z / abs(mean_z) * 100 if abs(mean_z) > 0 else 0
        
        # 计算Z方向偏差分布
        z_deviations = center_z_values - mean_z
        
        results = {
            'frame_count': len(center_z_values),
            'mean_z': mean_z,
            'std_z': std_z,
            'max_deviation_z': max_deviation_z,
            'min_deviation_z': min_deviation_z,
            'relative_error_z': relative_error_z,
            'z_values': center_z_values.tolist(),
            'z_deviations': z_deviations.tolist(),
            'frame_numbers': frame_numbers
        }
        
        print("\n=== 中心点Z方向重复定位精度 ===")
        print(f"测试帧数: {results['frame_count']}")
        print(f"平均Z坐标: {mean_z:.3f} mm")
        print(f"Z方向标准差: {std_z:.3f} mm")
        print(f"Z方向最大偏差: {max_deviation_z:.3f} mm")
        print(f"Z方向最小偏差: {min_deviation_z:.3f} mm")
        print(f"Z方向相对误差: {relative_error_z:.2f}%")
        
        return results
    
    def calculate_selected_points_precision(self) -> Dict:
        """计算选择点Z方向重复定位精度"""
        if self.data is None or not all(col in self.data.columns for col in ['Depth_mm', 'Point_ID']):
            return {'error': '数据格式不正确，需要Z、Point_ID列'}
        
        # 按Point_ID分组计算Z方向精度
        point_precisions = []
        point_numbers = []
        
        for point_id in self.data['Point_ID'].unique():
            point_data = self.data[self.data['Point_ID'] == point_id]
            if len(point_data) > 1:  # 至少需要2个数据点
                point_z_std = point_data['Depth_mm'].std()
                point_z_mean = point_data['Depth_mm'].mean()
                point_z_max_deviation = np.max(np.abs(point_data['Depth_mm'].values - point_z_mean))
                point_precisions.append({
                    'point_id': point_id,
                    'mean_z': point_z_mean,
                    'std_z': point_z_std,
                    'max_deviation_z': point_z_max_deviation,
                    'frame_count': len(point_data),
                    'z_values': point_data['Depth_mm'].values.tolist()
                })
                point_numbers.append(point_id)
        
        if not point_precisions:
            return {'error': '没有足够的点数据进行分析'}
        
        # 计算统计指标
        all_z_stds = np.array([p['std_z'] for p in point_precisions])
        all_z_max_deviations = np.array([p['max_deviation_z'] for p in point_precisions])
        
        mean_z_precision = np.mean(all_z_stds)
        std_z_precision = np.std(all_z_stds)
        mean_max_deviation_z = np.mean(all_z_max_deviations)
        
        # 找出Z方向精度最差和最好的点
        worst_point = max(point_precisions, key=lambda x: x['max_deviation_z'])
        best_point = min(point_precisions, key=lambda x: x['max_deviation_z'])
        
        results = {
            'point_count': len(point_precisions),
            'mean_z_precision': mean_z_precision,
            'std_z_precision': std_z_precision,
            'mean_max_deviation_z': mean_max_deviation_z,
            'worst_point': {
                'point_id': worst_point['point_id'],
                'max_deviation_z': worst_point['max_deviation_z'],
                'std_z': worst_point['std_z']
            },
            'best_point': {
                'point_id': best_point['point_id'],
                'max_deviation_z': best_point['max_deviation_z'],
                'std_z': best_point['std_z']
            },
            'point_details': point_precisions
        }
        
        print("\n=== 选择点Z方向重复定位精度 ===")
        print(f"分析点数: {results['point_count']}")
        print(f"Z方向平均精度: {mean_z_precision:.3f} mm")
        print(f"Z方向精度标准差: {std_z_precision:.3f} mm")
        print(f"Z方向平均最大偏差: {mean_max_deviation_z:.3f} mm")
        print(f"Z方向精度最差点: 点{worst_point['point_id']}, 最大偏差={worst_point['max_deviation_z']:.3f} mm")
        print(f"Z方向精度最好点: 点{best_point['point_id']}, 最大偏差={best_point['max_deviation_z']:.3f} mm")
        
        return results
    
    def generate_plots(self, output_dir: str = "precision_analysis"):
        """生成Z方向分析图表"""
        if self.data is None:
            print("没有数据可生成图表")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 中心点Z方向轨迹图
        if 'z_values' in self.results:
            z_values = np.array(self.results['z_values'])
            frame_numbers = self.results['frame_numbers']
            
            plt.figure(figsize=(12, 6))
            if len(z_values) > 1:
                plt.plot(frame_numbers, z_values, 'bo-', alpha=0.7, linewidth=2, markersize=6)
            else:
                plt.scatter(frame_numbers, z_values, s=100, alpha=0.7, color='blue')
            plt.axhline(y=self.results['mean_z'], color='r', linestyle='--', alpha=0.8, label=f'平均值: {self.results["mean_z"]:.3f} mm')
            plt.xlabel('Measurement_Index')
            plt.ylabel('Z坐标 (mm)')
            plt.title('中心点Z方向轨迹')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / 'center_point_z_trajectory.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Z方向偏差分布图
        if 'z_deviations' in self.results:
            z_deviations = np.array(self.results['z_deviations'])
            
            plt.figure(figsize=(10, 6))
            if len(z_deviations) > 1:
                plt.hist(z_deviations, bins=min(20, len(z_deviations)), alpha=0.7, edgecolor='black', color='skyblue')
            else:
                plt.bar([0], [1], alpha=0.7, edgecolor='black', color='skyblue', width=0.1)
                plt.text(0, 1, f'{z_deviations[0]:.3f}', ha='center', va='bottom')
            plt.xlabel('Z方向偏差 (mm)')
            plt.ylabel('频次')
            plt.title('Z方向偏差分布')
            plt.grid(True, alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            plt.tight_layout()
            plt.savefig(output_path / 'z_deviation_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 选择点Z方向精度对比图
        if 'point_details' in self.results:
            point_details = self.results['point_details']
            point_ids = [p['point_id'] for p in point_details]
            max_deviations_z = [p['max_deviation_z'] for p in point_details]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(point_ids, max_deviations_z, alpha=0.7, edgecolor='black', color='lightcoral')
            plt.xlabel('Point_ID')
            plt.ylabel('Z方向最大偏差 (mm)')
            plt.title('各选择点Z方向最大偏差对比')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, max_deviations_z):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path / 'point_z_precision_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 选择点Z方向标准差对比图
        if 'point_details' in self.results:
            point_details = self.results['point_details']
            point_ids = [p['point_id'] for p in point_details]
            std_deviations_z = [p['std_z'] for p in point_details]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(point_ids, std_deviations_z, alpha=0.7, edgecolor='black', color='lightgreen')
            plt.xlabel('Point_ID')
            plt.ylabel('Z方向标准差 (mm)')
            plt.title('各选择点Z方向标准差对比')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, std_deviations_z):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path / 'point_z_std_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nZ方向分析图表已保存到: {output_path}")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成Z方向分析报告"""
        if output_file is None:
            output_file = f"z_precision_analysis_report_{int(time.time())}.md"
        else:
            output_file = str(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Excel表格Z方向重复定位精度分析报告\n\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.excel_file}\n\n")
            
            # 数据结构
            if 'data_structure' in self.results:
                f.write("## 数据结构\n\n")
                f.write(f"- 总行数: {self.results['data_structure']['total_rows']}\n")
                f.write(f"- 帧数: {self.results['data_structure']['frame_count']}\n")
                f.write(f"- 每帧点数: {self.results['data_structure']['point_count']}\n\n")
            
            # 中心点Z方向精度
            if 'center_point' in self.results:
                center_result = self.results['center_point']
                f.write("## 中心点Z方向重复定位精度\n\n")
                f.write(f"- 测试帧数: {center_result['frame_count']}\n")
                f.write(f"- 平均Z坐标: {center_result['mean_z']:.3f} mm\n")
                f.write(f"- Z方向标准差: {center_result['std_z']:.3f} mm\n")
                f.write(f"- Z方向最大偏差: {center_result['max_deviation_z']:.3f} mm\n")
                f.write(f"- Z方向最小偏差: {center_result['min_deviation_z']:.3f} mm\n")
                f.write(f"- Z方向相对误差: {center_result['relative_error_z']:.2f}%\n\n")
            
            # 选择点Z方向精度
            if 'selected_points' in self.results:
                points_result = self.results['selected_points']
                f.write("## 选择点Z方向重复定位精度\n\n")
                f.write(f"- 分析点数: {points_result['point_count']}\n")
                f.write(f"- Z方向平均精度: {points_result['mean_z_precision']:.3f} mm\n")
                f.write(f"- Z方向精度标准差: {points_result['std_z_precision']:.3f} mm\n")
                f.write(f"- Z方向平均最大偏差: {points_result['mean_max_deviation_z']:.3f} mm\n")
                f.write(f"- Z方向精度最差点: 点{points_result['worst_point']['point_id']}, 最大偏差={points_result['worst_point']['max_deviation_z']:.3f} mm\n")
                f.write(f"- Z方向精度最好点: 点{points_result['best_point']['point_id']}, 最大偏差={points_result['best_point']['max_deviation_z']:.3f} mm\n\n")
            
            # Z方向精度等级评估
            f.write("## Z方向精度等级评估\n\n")
            if 'center_point' in self.results:
                max_dev_z = self.results['center_point']['max_deviation_z']
                if max_dev_z < 0.1:
                    grade = "优秀"
                elif max_dev_z < 0.5:
                    grade = "良好"
                elif max_dev_z < 1.0:
                    grade = "一般"
                else:
                    grade = "需要改进"
                
                f.write(f"- Z方向重复定位精度等级: {grade}\n")
                f.write(f"- Z方向最大偏差: {max_dev_z:.3f} mm\n\n")
        
        print(f"Z方向分析报告已生成: {output_file}")
        return output_file
    
    def run_analysis(self) -> Dict:
        """运行Z方向完整分析"""
        print("开始Excel表格Z方向重复定位精度分析...")
        
        # 加载数据
        if not self.load_data():
            return {}
        
        # 分析数据结构
        self.results['data_structure'] = self.analyze_data_structure()
        
        # 计算中心点Z方向精度
        center_result = self.calculate_center_point_precision()
        if 'error' not in center_result:
            self.results['center_point'] = center_result
        
        # 计算选择点Z方向精度
        points_result = self.calculate_selected_points_precision()
        if 'error' not in points_result:
            self.results['selected_points'] = points_result
        
        # 生成Z方向图表
        self.generate_plots()
        
        # 生成Z方向报告
        self.generate_report()
        
        print("\nZ方向分析完成！")
        return self.results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Excel表格Z方向重复定位精度分析程序')
    parser.add_argument('--excel_file',  default='./test_results/repeatability_test_1751872689.xlsx',help='Excel文件路径')
    parser.add_argument('--output-dir', default='precision_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建计算器并运行分析
    calculator = ExcelPrecisionCalculator(args.excel_file)
    results = calculator.run_analysis()
    
    if results:
        print("\n=== Z方向分析摘要 ===")
        if 'center_point' in results:
            max_dev_z = results['center_point']['max_deviation_z']
            print(f"Z方向重复定位精度: {max_dev_z:.3f} mm")
        
        if 'selected_points' in results:
            mean_dev_z = results['selected_points']['mean_max_deviation_z']
            print(f"选择点Z方向平均精度: {mean_dev_z:.3f} mm")


if __name__ == "__main__":
    import time
    main() 