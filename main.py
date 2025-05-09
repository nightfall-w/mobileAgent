"""
@FileName：main.py
@Description：
@Author：baojun.wang
@Time：2025/2/28 19:59
"""
import json
import os
import time

import certifi
from PIL import Image
from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

from image_processor import draw_points, encode_image
from mobile_tool import MobileUse
from utils.logging import logger

# model_name = "qwen2.5-vl-72b-instruct"
# model_name = "qwen-vl-max-latest"
model_name = 'Qwen/Qwen2.5-VL-72B-Instruct'
# model_name = 'Qwen/Qwen2-VL-7B-Instruct'

os.environ['SSL_CERT_FILE'] = certifi.where()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key="sk-3c65566a2fab4108a6f4ed1997b1f297",
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='733b9f61-b464-4e62-8ca9-627d7ebc679b',  # ModelScope Token
)


def run(user_query: str, device: str, cfg=None):
    """
    运行任务
    :param user_query: 用户输入自然语言
    :param device: 手机设备号
    :param cfg: 拓展配置
    :return:
    """
    if cfg is None:
        cfg = {}
    mobile_use = MobileUse(device=device, cfg=cfg)  # 获取手机设备状态 实例化操作工具类
    system_message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(
                text="你是一个UI任务执行工具"
            )])
        ],
        functions=[mobile_use.function],
        lang="zh",
    )

    system_message = system_message[0].model_dump()
    messages = []
    resized_path, w_scale_factor, h_scale_factor = mobile_use.take_screenshot_and_save()
    for i in range(50):  # 发起20对话，间隔3s(等上一步的点击操作生效，比如点击打来新页面 需要时间加载)
        if i == 0:
            screenshot = None
            base64_image = encode_image(screenshot) if screenshot else None
            # 第一次发起询问
            current_messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": user_query},
                    ] if screenshot else [{"type": "text", "text": user_query}],
                }
            ]
        else:
            time.sleep(3)
            screenshot, h_scale_factor, w_scale_factor = mobile_use.take_screenshot_and_save()
            user_query = """上一步已完成，这是此时的手机界面，要继续怎么做？操作了还没变化是页面上有弹窗遮挡吗"""

            base64_image = encode_image(screenshot) if screenshot else None

            # Build messages
            current_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": user_query},
                    ] if screenshot else [{"type": "text", "text": user_query}],
                }
            ]
        messages.extend(current_messages)

        completion = client.chat.completions.create(
            model=model_name,
            # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            # response_format={"type": "json_object"},
        )

        output_text = completion.choices[0].message.content
        print(output_text)
        if output_text.count("<tool_call>") > 1:
            output_text = output_text.replace('<tool_call>', "", 1)
        if output_text.count("}}}") > 0:
            output_text = output_text.replace('}}}', "}}", 1)
        if "<tool_call>" not in output_text:
            logger.error("大模型输出结果不符合要求，重新询问")
            continue
        if not output_text.endswith("</tool_call>"):
            try:
                var = output_text.split('<tool_call>\n')[1].split('\n')[0]
                action = json.loads(var)
            except Exception:
                logger.error("大模型输出结果不符合要求，重新询问")
                continue
        else:
            action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

        messages.append({"role": "assistant", "content": output_text})
        # action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        print(action)
        if screenshot:
            img_instance = Image.open(screenshot)
        else:
            img_instance = None
        if action['arguments'].get("action") == "terminate":
            display_image = draw_points(img_instance, points=[])
            display_image.show()
            if action['arguments'].get("status") == "success":
                logger.info("任务执行完成，结束!")
                break
            else:
                logger.error("任务执行失败，结束!")
                break

        if action['arguments'].get("coordinate2"):
            display_image = draw_points(img_instance,
                                        [action['arguments']['coordinate'], action['arguments']['coordinate2']])
            display_image.show()
        elif action['arguments'].get("coordinate"):
            display_image = draw_points(img_instance, [action['arguments']['coordinate'], ])
            display_image.show()
        else:
            if screenshot:
                display_image = draw_points(img_instance, [])
                display_image.show()
        mobile_use.call(action['arguments'], w_scale_factor=w_scale_factor, h_scale_factor=h_scale_factor)


if __name__ == "__main__":
    # user_query = """打开小红书，搜索用户:乐多对我笑，进入他的主页，给他的第2个贴子点赞，然后发私信:hello啊"""
    # user_query = """打开微信，小程序打开瑞幸咖啡，加购1杯美式和1杯任意拿铁"""
    user_query = """打开抖音，给第一条视频点赞，给下一条评论：拍的不错"""
    run(user_query=user_query, device="6HJDU19822005857")
    # run(user_query=user_query, device="bf75a03")
