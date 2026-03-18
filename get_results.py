from PIL import Image, ImageDraw, ImageFont
import os
import json

def _load_font(font_size=20):
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        if os.path.isabs(font_path) and not os.path.exists(font_path):
            continue
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw, text, font):
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_text_by_width(text, draw, font, max_width):
    if not text:
        return []

    wrapped_lines = []
    for raw_line in str(text).splitlines():
        if not raw_line:
            wrapped_lines.append("")
            continue

        current_line = ""
        for ch in raw_line:
            candidate = current_line + ch
            if _text_width(draw, candidate, font) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    wrapped_lines.append(current_line.rstrip())
                current_line = ch
        if current_line:
            wrapped_lines.append(current_line.rstrip())

    return wrapped_lines if wrapped_lines else [str(text)]


def merge_with_last_spaced(
    image_paths,
    output_path,
    spacing=10,
    last_spacing=50,
    background_color=(255, 255, 255),
    instruction_text=None,
    text_margin=40,
    text_padding=40,
    font_size=None,
):
    """
    水平排列图片，最后一张有更大的间距
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出图片路径
        spacing: 普通图片间距
        last_spacing: 最后一张图与前面的间距
        background_color: 背景颜色
    """
    # 打开所有图片
    images = [Image.open(img_path) for img_path in image_paths]
    
    # 计算总宽度（考虑不同的间距）
    total_width = sum(img.width for img in images)
    
    # 前面的图片间距
    if len(images) > 1:
        total_width += spacing * (len(images) - 2)  # 前面图片之间的间距
        total_width += last_spacing  # 最后一张图的额外间距
    
    # 最大高度
    max_height = max(img.height for img in images)
    
    if font_size is None:
        font_size = max(180, max_height // 3)
    font = _load_font(font_size=font_size)
    text_lines = []
    text_block_height = 0
    line_height = 0
    line_spacing = 0

    if instruction_text:
        probe_draw = ImageDraw.Draw(Image.new('RGB', (1, 1), background_color))
        max_text_width = max(1, total_width - 2 * text_margin)
        text_lines = _wrap_text_by_width(
            f"Instruction: {instruction_text}",
            probe_draw,
            font,
            max_text_width,
        )
        text_bbox = probe_draw.textbbox((0, 0), "Ag", font=font)
        line_height = text_bbox[3] - text_bbox[1]
        line_spacing = max(20, line_height // 2)
        text_block_height = (
            text_padding * 2
            + len(text_lines) * line_height
            + max(0, len(text_lines) - 1) * line_spacing
        )

    # 创建新图片
    canvas_height = max_height + text_block_height
    new_image = Image.new('RGB', (total_width, canvas_height), background_color)

    # 粘贴图片
    x_offset = 0
    for i, img in enumerate(images):
        # 计算垂直居中位置
        y_offset = text_block_height + (max_height - img.height) // 2
        
        # 粘贴图片
        new_image.paste(img, (x_offset, y_offset))
        
        # 更新x偏移量（最后一张图之前使用更大的间距）
        if i < len(images) - 1:
            if i >= len(images) - 2:  # 倒数第二张，后面使用大间距
                x_offset += img.width + last_spacing
            else:  # 其他图片使用普通间距
                x_offset += img.width + spacing
    
    if text_lines:
        draw = ImageDraw.Draw(new_image)
        draw.rectangle([0, 0, total_width, text_block_height], fill=(0, 0, 0))
        y_text = text_padding
        for line in text_lines:
            draw.text(
                (text_margin, y_text),
                line,
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            y_text += line_height + line_spacing

    # 保存图片
    new_image.save(output_path)
    print(f"图片已保存到: {output_path}")
    return new_image

if __name__ == "__main__":
    name = 'complex_2context_0310_3data_geo_aux_0310_0015000_complex_2context_0310_scannet_test'
    image_root = '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data'
    json_path = '/mnt/petrelfs/linjingli/UMM_Spatial/annotations/complex_2context_0310_scannet_test.jsonl'

    tmp_dir = f'/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/{name}'
    outdir = f'/mnt/petrelfs/linjingli/UMM_Spatial/bagel/ex_img/{name}'
    os.makedirs(outdir,exist_ok=True)
    
    
    def read_jsonl(file_path):
        """逐行读取JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data

    val_datas = read_jsonl(json_path)
    
    
    for image_file in os.listdir(tmp_dir):
        id_ = int(image_file.split('.')[0])
        matched_item = None
        for item in val_datas:
            if item['id'] == id_:
                matched_item = item
                break
        if matched_item is None:
            continue

        available_images = [os.path.join(image_root, t) for t in matched_item['context'] + [matched_item['target']]]
        available_images += [os.path.join(tmp_dir, image_file)]

        result = merge_with_last_spaced(
            available_images, 
            os.path.join(outdir,image_file),
            spacing=20,      
            last_spacing=80,
            instruction_text=matched_item.get('instruction', '')
        )
