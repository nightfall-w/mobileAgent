import base64

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor

from utils.config import ConfigParser
from utils.logging import logger


def encode_image(screenshot):
    return base64.b64encode(open(screenshot, "rb").read()).decode("utf-8")


def resize_image(input_image_path, output_image_path,
                 scale_factor=float(ConfigParser.get_config('screenshot', 'scale_factor'))):
    """
    重置图片大小 等比例缩放
    :param input_image_path: 原图路径
    :param output_image_path: 缩放后的截图路径
    :param scale_factor: 缩放比例
    :return:
    """
    try:
        # 打开图片
        image = Image.open(input_image_path)
        # 获取原始图片的宽度和高度
        width, height = image.size
        # 计算缩小后的宽度和高度
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # 调整图片大小
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # 保存调整后的图片
        resized_image.save(output_image_path)
        draw_rectangle(output_image_path)  # 绘制元素矩形框
        logger.info(f"图片已成功缩放，保存路径为: {output_image_path}")
    except Exception as e:
        logger.error(f"处理图片时出现错误: {e}")


def draw_points(image: Image.Image, points: list, inner_color=None, outer_color=None):
    """
    在图片上绘制目标操作点位
    :param image: 截图对象
    :param points: 点位列表
    :param inner_color: 内圈颜色
    :param outer_color: 外圈颜色
    :return:
    """
    if isinstance(inner_color, str):
        try:
            inner_color = ImageColor.getrgb(inner_color)
            inner_color = inner_color + (128,)
        except ValueError:
            inner_color = (255, 0, 0, 128)  # 默认红色
    else:
        inner_color = (255, 0, 0, 128)  # 默认红色

    if isinstance(outer_color, str):
        try:
            outer_color = ImageColor.getrgb(outer_color)
            outer_color = outer_color + (128,)
        except ValueError:
            outer_color = (0, 0, 255, 128)  # 默认蓝色
    else:
        outer_color = (0, 0, 255, 128)  # 默认蓝色
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    inner_radius = radius * 0.5
    out_pull_back_ratio = float(ConfigParser.get_config('screenshot', 'out_pull_back_ratio'))
    for point in points:
        x, y = point
        if x >= image.width - 50:
            x = image.width - image.width * out_pull_back_ratio
            logger.error(f"超出图片x边界，已自动拉回{image.width * out_pull_back_ratio}px")
        if y >= image.height - 50:
            y = image.height - image.height * out_pull_back_ratio
            logger.error(f"超出图片y边界，已自动拉回{image.height * out_pull_back_ratio}px")
        # 绘制外环
        overlay_draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=outer_color
        )
        # 绘制内环
        overlay_draw.ellipse(
            [(x - inner_radius, y - inner_radius), (x + inner_radius, y + inner_radius)],
            fill=inner_color
        )
    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    return combined.convert('RGB')


def preprocess_image(image):
    # 去噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 增强对比度
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    processed_image = cv2.merge((cl, a, b))
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)
    return processed_image


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


def filter_contours(contours, min_area=100, max_aspect_ratio=5):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / float(h) if h != 0 else float('inf')
        if area > min_area and aspect_ratio < max_aspect_ratio:
            filtered_contours.append(contour)
    return filtered_contours


def draw_rectangle(img_path: str):
    # 读取图像
    image = cv2.imread(img_path)
    # 预处理图像
    processed_image = preprocess_image(image)
    # 进行边缘检测
    edges = detect_edges(processed_image)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 定义不同颜色的框
    colors = [
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 红色
        (255, 0, 0),  # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 粉色
        (255, 255, 0)  # 青色
    ]
    # 过滤轮廓
    filtered_contours = filter_contours(contours)
    # 遍历轮廓，绘制矩形框
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color = colors[i % len(colors)]  # 循环使用颜色
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)  # 线条宽度为1
    cv2.imwrite(img_path, image)
    # # 显示标注后的图像
    # cv2.imshow('Annotated Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 保存标注后的图像
