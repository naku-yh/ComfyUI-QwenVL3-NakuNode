"""
Qwen3VL 对比打标节点
用于对比原始图片和AI修图结果，生成修图prompt的反推描述

功能特点：
- 支持双文件夹输入（原始图A文件夹 + 结果图B文件夹）
- 自动检查文件名一一对应关系
- 支持中英文提示词切换
- 可选择输出位置（默认B文件夹或自定义位置）
- 专门用于Kontext和Qwen等AI修图工具的prompt反推
- 基于现有批量打标节点的成熟架构

使用场景：
- AI修图工具的prompt反推分析
- 图像对比处理的自动化描述
- 修图效果的文字化记录
"""

import os
import sys
import time
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
import comfy.utils

# 动态导入同目录下的 qwen3vl_node 模块
try:
    from .qwen3vl_node import (
        Qwen3VL_Advanced, 
        MODEL_CONFIGS, 
        SYSTEM_PROMPTS,
        Quantization,
        ImageProcessor
    )
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import qwen3vl_node
    Qwen3VL_Advanced = qwen3vl_node.Qwen3VL_Advanced
    MODEL_CONFIGS = qwen3vl_node.MODEL_CONFIGS
    SYSTEM_PROMPTS = qwen3vl_node.SYSTEM_PROMPTS
    Quantization = qwen3vl_node.Quantization
    ImageProcessor = qwen3vl_node.ImageProcessor

# 内置的对比打标提示词
COMPARE_PROMPTS = {
    "中文": "第二张图片是根据第一张图片经过AI的修图得来的，你现在需要分析第二张图片相比于第一张改动了哪些内容，我需要反推出AI的prompt。这种prompt是指令式的，我需要你用自然语言描述输出这种指令式的结果主要分析图片是哪里做了变动，直接输出你的prompt结果，不要带任何解释性的文字",
    "English": "The second image is the result of AI-based photo editing applied to the first image. You need to analyze what changes were made in the second image compared to the first one, and reverse-engineer the AI prompt. This prompt should be instructional. I need you to describe in natural language the instructional result, mainly analyzing where the image was modified. Output your prompt result directly without any explanatory text."
}


class Qwen3VL_Compare_Caption:
    """Qwen3VL 对比打标节点 - 用于AI修图前后对比分析"""
    
    def __init__(self):
        # 复用 Qwen3VL_Advanced 的核心功能
        self.advanced_node = Qwen3VL_Advanced()
        self.last_seed = -1
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义对比打标节点输入类型"""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]

        return {
            "required": {
                "模型选择": (model_names, {"default": default_model}),
                "量化级别": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "A文件夹(原始图)": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入原始图片文件夹路径"
                }),
                "B文件夹(结果图)": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入AI修图结果文件夹路径"
                }),
                "语言选择": (["中文", "English"], {"default": "中文"}),
                "自定义提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "留空使用内置提示词，输入内容将覆盖内置提示词"
                }),
                "输出位置": (["默认(B文件夹)", "自定义位置"], {"default": "默认(B文件夹)"}),
                "自定义输出文件夹": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "选择自定义位置时使用"
                }),
                "最大令牌数": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "采样温度": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "核采样参数": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "开启TF32加速": ("BOOLEAN", {"default": True, "tooltip": "启用TF32加速（仅支持Ampere及以上架构显卡，如30/40/50系，能显著提升速度）"}),
                "保持模型加载": ("BOOLEAN", {"default": True}),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，-1为随机"
                }),
                "种子控制": (["随机", "固定", "递增"], {"default": "随机"}),
                "前缀文本": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "添加到描述开头的文本"
                }),
                "后缀文本": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "添加到描述结尾的文本"
                }),
                "强制覆盖": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后会覆盖已存在的txt文件"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("处理结果",)
    FUNCTION = "compare_process"
    CATEGORY = "NakuNode-QWen3VL"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed_control = kwargs.get("种子控制", "随机")
        seed = kwargs.get("随机种子", -1)

        # 随机和递增模式下，强制更新 (返回 NaN)
        if seed_control in ["随机", "递增"]:
            return float("nan")

        # 固定模式下，仅当种子值变化时更新
        return seed
    
    def process_image_pair(self, image_a_path: str, image_b_path: str, prompt_text: str, **kwargs) -> str:
        """
        处理一对图像（原始图和结果图）
        
        Args:
            image_a_path: 原始图像文件路径
            image_b_path: 结果图像文件路径
            prompt_text: 提示词文本
            **kwargs: 其他参数
            
        Returns:
            生成的对比分析文本
        """
        # 移除try-except，让错误直接抛出
        print(f"   [开始] 开始处理图像对: {os.path.basename(image_a_path)} vs {os.path.basename(image_b_path)}")
        
        # 加载两张图像
        images = []
        for img_path in [image_a_path, image_b_path]:
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 转换为tensor格式 (ComfyUI格式: H,W,C, 范围0-1)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # 添加batch维度
                images.append(img_tensor)
        
        # 调用高级节点的处理函数，传入两张图像
        result = self.advanced_node.process(
            **{
                "模型选择": kwargs.get("模型名称"),
                "量化级别": kwargs.get("量化级别"),
                "预设提示词": "自定义",  # 使用自定义提示词
                "自定义提示词": prompt_text,
                "最大令牌数": kwargs.get("最大令牌数"),
                "采样温度": kwargs.get("采样温度"),
                "核采样参数": kwargs.get("核采样参数"),
                "束搜索数量": 1,
                "重复惩罚": 1.2,
                "视频帧数": 16,
                "设备选择": "auto",
                "开启TF32加速": kwargs.get("开启TF32加速", False),
                "保持模型加载": True,  # 批量处理时始终保持加载
                "随机种子": kwargs.get("随机种子"),
                "种子控制": kwargs.get("种子控制"),
                "图像1": images[0],  # 原始图（第一张）
                "图像2": images[1],  # 结果图（第二张）
            }
        )
        
        return result[0] if result else ""
    
    def check_file_correspondence(self, folder_a: str, folder_b: str):
        """
        检查两个文件夹中的文件是否一一对应
        
        Args:
            folder_a: A文件夹路径
            folder_b: B文件夹路径
            
        Returns:
            (是否对应, 对应的文件列表, 错误信息)
        """
        # 支持的图像格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
        
        # 收集A文件夹的图像文件
        files_a = []
        if os.path.exists(folder_a):
            for filename in os.listdir(folder_a):
                if filename.lower().endswith(image_extensions):
                    files_a.append(filename)
        
        # 收集B文件夹的图像文件
        files_b = []
        if os.path.exists(folder_b):
            for filename in os.listdir(folder_b):
                if filename.lower().endswith(image_extensions):
                    files_b.append(filename)
        
        # 排序文件列表
        files_a.sort()
        files_b.sort()
        
        # 检查数量是否一致
        if len(files_a) != len(files_b):
            return False, [], f"文件数量不匹配: A文件夹({len(files_a)}个) vs B文件夹({len(files_b)}个)"
        
        if len(files_a) == 0:
            return False, [], "两个文件夹都没有找到图像文件"
        
        # 检查文件名是否一一对应（忽略扩展名）
        mismatched_files = []
        matched_pairs = []
        
        for i, (file_a, file_b) in enumerate(zip(files_a, files_b)):
            base_a = os.path.splitext(file_a)[0]
            base_b = os.path.splitext(file_b)[0]
            
            if base_a == base_b:
                matched_pairs.append((file_a, file_b))
            else:
                mismatched_files.append(f"位置{i+1}: {file_a} vs {file_b}")
        
        if mismatched_files:
            error_msg = f"文件名不匹配:\n" + "\n".join(mismatched_files)
            return False, [], error_msg
        
        return True, matched_pairs, ""
    
    @torch.no_grad()
    def compare_process(self, **kwargs):
        """对比处理文件夹中的图像对"""
        # 提取参数
        模型名称 = kwargs.get("模型选择")
        量化级别 = kwargs.get("量化级别")
        A文件夹 = kwargs.get("A文件夹(原始图)", "").strip()
        B文件夹 = kwargs.get("B文件夹(结果图)", "").strip()
        语言选择 = kwargs.get("语言选择", "中文")
        自定义提示词 = kwargs.get("自定义提示词", "").strip()
        输出位置 = kwargs.get("输出位置", "默认(B文件夹)")
        自定义输出文件夹 = kwargs.get("自定义输出文件夹", "").strip()
        最大令牌数 = kwargs.get("最大令牌数")
        采样温度 = kwargs.get("采样温度")
        核采样参数 = kwargs.get("核采样参数")
        保持模型加载 = kwargs.get("保持模型加载")
        开启TF32加速 = kwargs.get("开启TF32加速", False)
        随机种子 = kwargs.get("随机种子")
        种子控制 = kwargs.get("种子控制", "随机")
        前缀文本 = kwargs.get("前缀文本", "").strip()
        后缀文本 = kwargs.get("后缀文本", "").strip()
        强制覆盖 = kwargs.get("强制覆盖", False)
        
        # 设置 TF32 加速
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = 开启TF32加速
            torch.backends.cudnn.allow_tf32 = 开启TF32加速
            if 开启TF32加速:
                print("[加速] 已开启 TF32 加速模式")

        # 如果种子控制是随机，默认强制覆盖
        if 种子控制 == "随机":
            强制覆盖 = True
        
        # 验证输入文件夹
        if not A文件夹 or not os.path.exists(A文件夹):
            return ("❌ 错误: A文件夹(原始图)路径无效或不存在",)
        
        if not B文件夹 or not os.path.exists(B文件夹):
            return ("❌ 错误: B文件夹(结果图)路径无效或不存在",)
        
        # 确定输出文件夹
        if 输出位置 == "默认(B文件夹)":
            输出文件夹 = B文件夹
        else:
            if not 自定义输出文件夹:
                return ("❌ 错误: 选择自定义位置时必须指定输出文件夹",)
            输出文件夹 = 自定义输出文件夹
            os.makedirs(输出文件夹, exist_ok=True)
        
        # 确定使用的提示词
        if 自定义提示词:
            prompt_text = 自定义提示词
        else:
            prompt_text = COMPARE_PROMPTS.get(语言选择, COMPARE_PROMPTS["中文"])
        
        # 检查文件对应关系
        is_matched, file_pairs, error_msg = self.check_file_correspondence(A文件夹, B文件夹)
        
        if not is_matched:
            return (f"❌ 文件对应检查失败: {error_msg}",)
        
        print(f"\n{'='*60}")
        print(f"[开始] 开始对比打标")
        print(f"[A文件夹] A文件夹(原始图): {A文件夹}")
        print(f"[B文件夹] B文件夹(结果图): {B文件夹}")
        print(f"[输出] 输出文件夹: {输出文件夹}")
        print(f"[图像] 图像对数量: {len(file_pairs)}")
        print(f"[语言] 语言选择: {语言选择}")
        print(f"[提示词] 使用提示词: {'自定义' if 自定义提示词 else '内置'}")
        print(f"[种子] 种子控制: {种子控制}")
        print(f"[覆盖] 强制覆盖: {'是' if 强制覆盖 else '否'}")
        print(f"[文件] 找到的图像对:")
        for i, (file_a, file_b) in enumerate(file_pairs, 1):
            print(f"   {i}. {file_a} ↔ {file_b}")
        print(f"{'='*60}\n")
        
        # 统计信息
        成功数量 = 0
        失败数量 = 0
        跳过数量 = 0
        开始时间 = time.time()
        
        # 记录处理结果的详细信息
        成功文件 = []
        失败文件 = []
        跳过文件 = []
        
        # 创建进度条
        pbar = comfy.utils.ProgressBar(len(file_pairs))
        
        # 预先尝试加载模型，以便在模型缺失时立即弹出错误（不被try-except捕获）
        self.advanced_node.load_model(模型名称, 量化级别, "auto")
        
        # 计算有效的初始种子
        if 种子控制 == "固定":
            effective_seed = 随机种子 if 随机种子 != -1 else random.randint(0, 2147483647)
        elif 种子控制 == "随机":
            effective_seed = random.randint(0, 2147483647)
        elif 种子控制 == "递增":
            if self.last_seed == -1:
                effective_seed = 随机种子 if 随机种子 != -1 else random.randint(0, 2147483647)
            else:
                effective_seed = self.last_seed + 1
        else:
            effective_seed = random.randint(0, 2147483647)
            
        self.last_seed = effective_seed
        
        # 处理每对图像
        for idx, (file_a, file_b) in enumerate(file_pairs):
            # 移除try以便错误能直接抛出
            image_a_path = os.path.join(A文件夹, file_a)
            image_b_path = os.path.join(B文件夹, file_b)
            
            # 确定输出文件名（基于B文件夹的文件名）
            base_name = os.path.splitext(file_b)[0]
            text_path = os.path.join(输出文件夹, f"{base_name}.txt")
            
            # 检查是否已存在描述文件（只有在不强制覆盖时才跳过）
            if os.path.exists(text_path) and not 强制覆盖:
                print(f"[跳过] 跳过已存在: {file_b} (如需覆盖请启用'强制覆盖'选项)")
                跳过数量 += 1
                跳过文件.append(f"{file_a} ↔ {file_b}")
                pbar.update_absolute(idx + 1, len(file_pairs))
                continue
            elif os.path.exists(text_path) and 强制覆盖:
                print(f"[覆盖] 强制覆盖: {file_b}")

            print(f"[处理] 处理中 [{idx+1}/{len(file_pairs)}]: {file_a} ↔ {file_b}")
            print(f"   [路径] 原始图路径: {image_a_path}")
            print(f"   [路径] 结果图路径: {image_b_path}")
            print(f"   [输出] 输出路径: {text_path}")
            
            # 处理图像对
            caption = self.process_image_pair(
                image_a_path,
                image_b_path,
                prompt_text,
                模型名称=模型名称,
                量化级别=量化级别,
                最大令牌数=最大令牌数,
                采样温度=采样温度,
                核采样参数=核采样参数,
                开启TF32加速=开启TF32加速,
                随机种子=随机种子,
                种子控制=种子控制
            )
            
            if caption:
                # 添加前缀和后缀
                if 前缀文本:
                    caption = f"{前缀文本} {caption}"
                if 后缀文本:
                    caption = f"{caption} {后缀文本}"
                
                # 保存描述文件
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                print(f"[成功] 成功: {base_name}.txt")
                print(f"   描述: {caption[:100]}{'...' if len(caption) > 100 else ''}\n")
                成功数量 += 1
                成功文件.append(f"{file_a} ↔ {file_b}")
            else:
                print(f"[失败] 失败: 生成描述为空\n")
                失败数量 += 1
                失败文件.append(f"{file_a} ↔ {file_b}")
            
            # 更新进度条
            pbar.update_absolute(idx + 1, len(file_pairs))
        
        # 清理模型（如果不保持加载）
        if not 保持模型加载:
            self.advanced_node.clear_model_resources()
        
        # 计算总耗时
        总耗时 = time.time() - 开始时间
        平均耗时 = 总耗时 / len(file_pairs) if file_pairs else 0
        
        # 生成详细的结果报告
        详细报告 = []

        if 成功文件:
            详细报告.append("[成功] 成功处理的图像对:")
            for file_pair in 成功文件:
                详细报告.append(f"   • {file_pair}")

        if 失败文件:
            详细报告.append("[失败] 处理失败的图像对:")
            for file_pair in 失败文件:
                详细报告.append(f"   • {file_pair}")

        if 跳过文件:
            详细报告.append("[跳过] 跳过的图像对:")
            for file_pair in 跳过文件:
                详细报告.append(f"   • {file_pair}")
        
        详细信息 = "\n".join(详细报告) if 详细报告 else ""
        
        # 生成结果报告
        结果报告 = f"""
{'='*60}
[完成] 对比打标完成报告
{'='*60}
[统计] 处理统计:
   • 总图像对数: {len(file_pairs)}
   • 成功处理: {成功数量}
   • 处理失败: {失败数量}
   • 跳过文件: {跳过数量}

[耗时] 时间统计:
   • 总耗时: {总耗时:.2f} 秒
   • 平均耗时: {平均耗时:.2f} 秒/对

[文件夹] 文件夹信息:
   • A文件夹(原始图): {A文件夹}
   • B文件夹(结果图): {B文件夹}
   • 输出文件夹: {输出文件夹}

[配置] 处理配置:
   • 语言选择: {语言选择}
   • 提示词类型: {'自定义' if 自定义提示词 else '内置'}
   • 强制覆盖: {'是' if 强制覆盖 else '否'}

{详细信息}
{'='*60}
"""
        
        print(结果报告)
        return (结果报告,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Compare_Caption": Qwen3VL_Compare_Caption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Compare_Caption": "NakuNode-QWen3VL对比打标",
}
