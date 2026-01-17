"""
Qwen3VL 额外选项节点
用于配置Qwen3VL的高级描述选项，可以连接到批量打标节点

功能特点：
- 提供Qwen3VL的所有高级描述选项配置
- 输出格式化的选项字典，可连接到其他节点
- 模块化设计，保持主节点的简洁性
- 支持精细化控制图像描述的生成内容和风格
"""

class Qwen3VL_ExtraOptions:
    """Qwen3VL 额外选项配置节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义Qwen3VL额外选项的输入类型"""
        return {
            "required": {},
            "optional": {
                # 人物信息控制
                "包含人物信息": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "如果图像中有人物/角色，包含相关信息（姓名等）"
                }),
                "排除不可改变特征": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "不包含无法改变的人物特征信息（如种族、性别等），但仍包含可改变的属性（如发型）"
                }),

                # 技术细节
                "包含光照信息": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含关于光照的信息"
                }),
                "包含相机角度": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含相机角度信息"
                }),
                "包含相机详情": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "如果是照片，必须包含使用的相机信息和详细信息（如光圈、快门速度、ISO等）"
                }),
                "提及光源": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "如果适用，提及可能使用的人工或自然光源"
                }),

                # 图像质量评估
                "包含艺术质量": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "必须包含关于图像美学/艺术质量的信息，从非常低到非常高"
                }),
                "包含构图信息": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含图像构图信息，如三分法、引导线、对称性等"
                }),
                "包含景深信息": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "指定景深和背景是否对焦或模糊"
                }),

                # 内容过滤
                "排除性感内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "不包含任何性感或暗示性内容"
                }),
                "不提及文字": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "不提及图像中的任何文字"
                }),
                "不提及分辨率": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "不提及图像的分辨率"
                }),

                # 技术信息
                "包含水印信息": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含图像是否有水印的信息"
                }),
                "包含JPEG伪影": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含图像是否有JPEG压缩伪影的信息"
                }),

                # 描述风格控制
                "不使用模糊语言": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "不使用模糊的语言"
                }),
                "描述重要元素": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅描述图像中最重要的元素"
                }),
                "包含安全性": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "包含图像是否安全、暗示性或不安全的信息"
                }),
            }
        }
    
    RETURN_TYPES = ("QWEN3VL_EXTRA_OPTIONS",)
    RETURN_NAMES = ("Qwen3VL额外选项",)
    FUNCTION = "create_options"
    CATEGORY = "NakuNode-QWen3VL"
    
    def create_options(self, **kwargs):
        """
        创建Qwen3VL额外选项字典
        
        Returns:
            包含所有选项的字典
        """
        # 提取所有选项参数
        options = {
            "包含人物信息": kwargs.get("包含人物信息", False),
            "排除不可改变特征": kwargs.get("排除不可改变特征", False),
            "包含光照信息": kwargs.get("包含光照信息", False),
            "包含相机角度": kwargs.get("包含相机角度", False),
            "包含相机详情": kwargs.get("包含相机详情", False),
            "提及光源": kwargs.get("提及光源", False),
            "包含艺术质量": kwargs.get("包含艺术质量", False),
            "包含构图信息": kwargs.get("包含构图信息", False),
            "包含景深信息": kwargs.get("包含景深信息", False),
            "排除性感内容": kwargs.get("排除性感内容", False),
            "不提及文字": kwargs.get("不提及文字", False),
            "不提及分辨率": kwargs.get("不提及分辨率", False),
            "包含水印信息": kwargs.get("包含水印信息", False),
            "包含JPEG伪影": kwargs.get("包含JPEG伪影", False),
            "不使用模糊语言": kwargs.get("不使用模糊语言", False),
            "描述重要元素": kwargs.get("描述重要元素", False),
            "包含安全性": kwargs.get("包含安全性", False),
        }
        
        # 统计启用的选项
        enabled_count = sum(1 for value in options.values() if value)
        
        print(f"[配置] Qwen3VL额外选项配置完成:")
        print(f"   启用选项数量: {enabled_count}")
        if enabled_count > 0:
            enabled_options = [key for key, value in options.items() if value]
            print(f"   启用的选项: {', '.join(enabled_options)}")
        
        return (options,)

    @staticmethod
    def build_enhanced_prompt(base_prompt: str, options: dict) -> str:
        """
        根据Qwen3VL额外选项构建增强的提示词
        
        Args:
            base_prompt: 基础提示词
            options: Qwen3VL额外选项字典
            
        Returns:
            增强后的提示词
        """
        enhanced_instructions = []
        
        # 根据选项添加具体指令
        if options.get("包含人物信息", False):
            enhanced_instructions.append("如果图像中有人物/角色，请包含相关信息（如姓名等）。")
        
        if options.get("排除不可改变特征", False):
            enhanced_instructions.append("不要包含无法改变的人物特征信息（如种族、性别等），但可以包含可改变的属性（如发型）。")
        
        if options.get("包含光照信息", False):
            enhanced_instructions.append("请描述图像的光照情况。")
        
        if options.get("包含相机角度", False):
            enhanced_instructions.append("请描述相机角度信息。")
        
        if options.get("包含相机详情", False):
            enhanced_instructions.append("如果是照片，请包含使用的相机信息和详细参数（如光圈、快门速度、ISO等）。")
        
        if options.get("提及光源", False):
            enhanced_instructions.append("如果适用，请提及可能使用的人工或自然光源。")
        
        if options.get("包含艺术质量", False):
            enhanced_instructions.append("请评价图像的美学/艺术质量（从非常低到非常高）。")
        
        if options.get("包含构图信息", False):
            enhanced_instructions.append("请描述图像构图信息，如三分法、引导线、对称性等。")
        
        if options.get("包含景深信息", False):
            enhanced_instructions.append("请描述景深和背景是否对焦或模糊。")
        
        if options.get("排除性感内容", False):
            enhanced_instructions.append("不要包含任何性感或暗示性内容的描述。")
        
        if options.get("不提及文字", False):
            enhanced_instructions.append("不要提及图像中的任何文字内容。")
        
        if options.get("不提及分辨率", False):
            enhanced_instructions.append("不要提及图像的分辨率。")
        
        if options.get("包含水印信息", False):
            enhanced_instructions.append("请说明图像是否有水印。")
        
        if options.get("包含JPEG伪影", False):
            enhanced_instructions.append("请说明图像是否有JPEG压缩伪影。")
        
        if options.get("不使用模糊语言", False):
            enhanced_instructions.append("请使用具体、准确的语言，避免模糊的表达。")
        
        if options.get("描述重要元素", False):
            enhanced_instructions.append("请重点描述图像中最重要的元素。")
        
        if options.get("包含安全性", False):
            enhanced_instructions.append("请评价图像是否安全、暗示性或不安全。")
        
        # 构建最终的提示词
        if enhanced_instructions:
            instructions_text = "\n".join([f"- {instruction}" for instruction in enhanced_instructions])
            enhanced_prompt = f"{base_prompt}\n\n请遵循以下额外要求：\n{instructions_text}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ExtraOptions": Qwen3VL_ExtraOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ExtraOptions": "NakuNode-QWen3VL额外选项",
}
