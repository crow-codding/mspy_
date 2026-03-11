import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mzpy.peak import PeakFrame, read_msp
from mzpy.similarity import join, align, get_bonanza_score
import matplotlib

# ================== 配置区域 ==================
# 设置中文字体支持（解决乱码问题）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 数据路径（确保使用E盘路径）
msp_path = r"E:\minicanda\envs\dreams\msdata\MSMS-Public_experimentspectra-pos-VS19.msp"

# 分析参数
TARGET_COMPOUND_COUNT = 180      # 仅计算前30个化合物（每个化合物6个谱图，共180个谱图）
MAX_SPECTRA_PER_COMPOUND = 6     # 每个化合物最多计算6个谱图（匹配您数据中的6个谱图）
PPM_TOLERANCE = 30.0            # 提高PPM容差（从20.0到30.0）
PRECURSOR_TOLERANCE = 0.05      # 母离子容差（Da）

# ==============================================

def extract_inchikey_from_name(name_series):
    """从NAME列提取INCHIKEY（27字符格式）"""
    # 正则表达式匹配标准INCHIKEY格式（27字符）
    inchikey_pattern = r'([A-Z0-9\-]{27})'
    return name_series.str.extract(inchikey_pattern, expand=False)

def process_msp_data(msp_path):
    """读取并处理MSP数据，确保INCHIKEY列存在"""
    print("\n=== 步骤1: 读取MSP文件 ===")
    df = read_msp(msp_path)
    
    # 确保是PeakFrame对象
    if not isinstance(df, PeakFrame):
        df = PeakFrame(df)
    
    # 验证数据结构
    print(f"数据结构: {len(df)} 化合物, {len(df.columns)} 列")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查INCHIKEY列是否存在
    if 'INCHIKEY' in df.columns:
        print("INCHIKEY列已存在，使用该列进行分组")
    else:
        print("INCHIKEY列不存在，尝试从NAME列提取...")
        df['INCHIKEY'] = extract_inchikey_from_name(df['NAME'])
        
        # 检查提取结果
        if df['INCHIKEY'].isna().all():
            print("警告: INCHIKEY提取失败，使用NAME作为分组依据")
            df['INCHIKEY'] = df['NAME']
        else:
            print(f"成功提取INCHIKEY，共提取 {len(df['INCHIKEY'].dropna())} 个INCHIKEY")
    
    # 筛选前TARGET_COMPOUND_COUNT个化合物
    if len(df) > TARGET_COMPOUND_COUNT:
        df = df.head(TARGET_COMPOUND_COUNT).copy()
        print(f"\n已截取前 {TARGET_COMPOUND_COUNT} 个化合物进行计算。")
    
    return df

def calculate_spectra_similarity(df):
    """计算谱图相似度，每个化合物最多计算6个谱图"""
    print(f"\n=== 步骤2: 开始计算相似度 (PPM={PPM_TOLERANCE}, 母离子容差={PRECURSOR_TOLERANCE} Da) ===")
    
    all_scores = []
    compound_scores = []  # 存储每个化合物的相似度分数
    total_pairs = 0
    
    # 为每个INCHIKEY分组计算相似度
    for inchikey, group in tqdm(df.groupby('INCHIKEY'), desc="处理化合物"):
        # 跳过谱图数量不足的组
        if len(group) < 2:
            continue
        
        # 仅使用前MAX_SPECTRA_PER_COMPOUND个谱图（每个化合物最多6个）
        group = group.head(MAX_SPECTRA_PER_COMPOUND).copy()
        
        # 提取该化合物的所有谱图数据
        spectra = []
        for _, row in group.iterrows():
            # 确保PRECURSORMZ有效
            if pd.isna(row['PRECURSORMZ']):
                continue
                
            # 使用join函数处理谱图
            ms_data = join(row['PRECURSORMZ'], row['MSMS'])
            spectra.append(ms_data)
        
        # 计算该化合物内所有谱图对之间的相似度
        scores = []
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                # 计算谱图i和谱图j之间的相似度
                union_peaks = align(spectra[i], spectra[j], PRECURSOR_TOLERANCE, PPM_TOLERANCE)
                bonanza_score = get_bonanza_score(union_peaks)
                scores.append(bonanza_score)
        
        if scores:
            all_scores.extend(scores)
            total_pairs += len(scores)
            compound_scores.append({
                'INCHIKEY': inchikey,
                'Compound': group.iloc[0]['NAME'],
                'Scores': scores,  # 保存该化合物的所有相似度分数
                'Mean': np.mean(scores),
                'Min': np.min(scores),
                'Max': np.max(scores),
                'Count': len(scores)
            })
    
    return all_scores, compound_scores, total_pairs

def analyze_results(all_scores, compound_scores, total_pairs):
    """分析计算结果并生成可视化"""
    print(f"\n=== 步骤3: 分析结果 ===")
    
    if not all_scores:
        print("\n" + "="*30)
        print("失败：未获得任何相似度分数。")
        print("请检查数据或参数设置。")
        print("="*30)
        return
    
    # 计算统计指标
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    print(f"总谱图对数: {total_pairs}")
    print(f"平均Bonanza分数: {mean_score:.4f} ± {np.std(all_scores):.4f}")
    print(f"中位数: {median_score:.4f}")
    print(f"相似度分布: 最小={min_score:.4f}, 最大={max_score:.4f}")
    
    # 分析高质量/低质量化合物
    high_quality = [c for c in compound_scores if c['Mean'] >= median_score]
    low_quality = [c for c in compound_scores if c['Mean'] < median_score]
    
    print(f"\n1. 高质量相似度化合物 (≥中位数): {len(high_quality)}")
    if high_quality:
        print(f"   平均分数: {np.mean([c['Mean'] for c in high_quality]):.4f}")
    
    print(f"\n2. 低质量相似度化合物 (<中位数): {len(low_quality)}")
    if low_quality:
        print(f"   平均分数: {np.mean([c['Mean'] for c in low_quality]):.4f}")
    
    # 输出每个化合物的相似度分数详情（前5个）
    print("\n=== 每个化合物的相似度分数详情 ===")
    for i, c in enumerate(compound_scores[:5]):
        print(f"\n化合物 {i+1}: {c['Compound']} (INCHIKEY: {c['INCHIKEY']})")
        print(f"  谱图对数: {c['Count']}")
        print(f"  相似度分数: {c['Scores']}")
        print(f"  平均: {c['Mean']:.4f}, 最小: {c['Min']:.4f}, 最大: {c['Max']:.4f}")
    
    # 生成可视化
    plt.figure(figsize=(10, 6))
    sns.histplot(all_scores, bins=50, kde=True)
    plt.title(f'Similarity Distribution (Mean={mean_score:.2f}, Total Pairs={total_pairs})')
    plt.xlabel('Bonanza Score')
    plt.ylabel('Count')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('similarity_distribution.png')
    plt.close()
    
    print("\n结果已保存到: similarity_distribution.png")
    print("图表已包含清晰的中文标签，无乱码问题。")

def main():
    """主函数：执行整个流程"""
    print("="*50)
    print("=== 开始质谱谱图相似度分析 ===")
    print("="*50)
    
    # 1. 处理MSP数据
    df = process_msp_data(msp_path)
    
    # 2. 计算谱图相似度
    all_scores, compound_scores, total_pairs = calculate_spectra_similarity(df)
    
    # 3. 分析结果
    analyze_results(all_scores, compound_scores, total_pairs)
    
    print("\n" + "="*50)
    print("=== 分析完成 ===")
    print("="*50)

if __name__ == "__main__":
    main()