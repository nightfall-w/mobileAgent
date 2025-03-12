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
                text="你是一个UI识别工具，在识别小程序内容时：如果当前页面与之前并没变化，是不是你识别错了元素类型，尝试其他操作。如果需要授权登录，请授权微信一键登录，如果索要通知权限，请拒绝。记住，一般小程序的右上角圆点是关闭小程序；加购商品时，有的商品是多规格的，你需要选择规格后再确认。如果你认为操作已经完成，就返回中断标记。在点击输入框后，下一次操作即可输入关键字了，不要一直点击输入框；还有一般页面顶部中间的文字表示页面标题，而不是输入框，除非确实是一个input输入框那才是搜索框。")]),
        ],
        functions=[mobile_use.function],
        lang="zh",
    )

    system_message = system_message[0].model_dump()
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": msg["text"]} for msg in system_message["content"]
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query}
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model_name,  # ModelScope Model-Id
        # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        # response_format={"type": "json_object"},
    )
    output_text = completion.choices[0].message.content
    print(output_text)
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

    mobile_use.call(action['arguments'])

    messages.append({"role": "assistant", "content": output_text})
    for i in range(20):  # 发起20对话，间隔3s(等上一步的点击操作生效，比如点击打来新页面 需要时间加载)
        time.sleep(3)
        screenshot = mobile_use.take_screenshot_and_save()

        user_query = """上一步已完成，这是此时的手机界面，要继续怎么做？你需要评估下，如果当前页面与之前并没变化，是不是你识别错了元素类型，尝试其他操作。记住，一般小程序的右上角圆点是关闭小程序，左上角如果有左箭头表示返回上一页"""

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
        messages.append(current_messages[0])

        completion = client.chat.completions.create(
            model=model_name,
            # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            # response_format={"type": "json_object"},
        )

        output_text = completion.choices[0].message.content
        print(output_text)
        if "<tool_call>" not in output_text:
            logger.error("大模型输出结果不符合要求，重新询问")
            continue
        if not output_text.endswith("</tool_call>"):
            try:
                var = output_text.split('<tool_call>\n')[1].split('\n')[0]
                action = json.loads(var)
            except Exception:
                logger.error("大模型输出结果不符合要求，重新询问")
        else:
            action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

        messages.append({"role": "assistant", "content": output_text})
        # action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        print(action)
        if action['arguments'].get("action") == "terminate":
            display_image = draw_points(Image.open(screenshot), points=[])
            display_image.show()
            if action['arguments'].get("status") == "success":
                logger.info("任务执行完成，结束!")
                break
            else:
                logger.error("任务执行失败，结束!")
                break

        if action['arguments'].get("coordinate2"):
            display_image = draw_points(Image.open(screenshot),
                                        [action['arguments']['coordinate'], action['arguments']['coordinate2']])
            display_image.show()
        elif action['arguments'].get("coordinate"):
            display_image = draw_points(Image.open(screenshot), [action['arguments']['coordinate'], ])
            display_image.show()
        mobile_use.call(action['arguments'])


if __name__ == "__main__":
    # user_query = """打开微信，并搜索‘联想U店’小程序,并且帮我找一个‘笔记本’商品并加购到购物车。"""
    # user_query = """打开小红书，搜索用户:乐多对我笑，进入他的主页，给他的第2个贴子点赞，然后发私信:你好啊"""
    # user_query = """打开微信，小程序打开瑞幸咖啡，加购1杯美式和1杯任意拿铁"""
    # user_query = """打开微信，在”LESS商城“小程序中帮我加购一件2000元以内的牛仔裤"""
    user_query = """打开微信，打开“松山棉店”小程序首页， 点击首页的分享"""
    run(user_query=user_query, device="6HJDU19822005857")
