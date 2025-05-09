import os
import subprocess
import time as tm
from typing import Union, Tuple, List

from qwen_agent.tools.base import BaseTool, register_tool

from app_package import AppPackage
from image_processor import resize_image
from utils.logging import logger
from utils.config import ConfigParser
from qwen_vl_utils import smart_resize


@register_tool("mobile_use")
class MobileUse(BaseTool):
    parameters = {
        "properties": {
            "action": {
                "description": """
                The action to perform. The available actions are:
                * `key`: Perform a key event on the mobile device.
                    - This supports adb's `keyevent` syntax.
                    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
                * `click`: Click the point on the screen with coordinate (x, y).
                * `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
                * `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
                * `type`: Input the specified text into the activated input box.
                * `system_button`: Press the system button.
                * `open`: Open an app on the device.
                * `wait`: Wait specified seconds for the change to happen.
                * `terminate`: Terminate the current task and report its completion status.
                """.strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, device: str = None, cfg: dict = None):
        if cfg is None:
            cfg = dict()
        if not self._is_device_online(device):
            raise ValueError(f"Device {device} is not online or does not exist.")

        self.device = device
        self.display_width_px, self.display_height_px = self.get_device_resolution()
        self.out_pull_back_ratio = float(ConfigParser.get_config('screenshot', 'out_pull_back_ratio'))

        super().__init__(cfg)

    def _is_device_online(self, device):
        result = subprocess.run("adb devices", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to check device status: {result.stderr}")

        devices = result.stdout.strip().split('\n')[1:]  # Skip the header line
        for dev in devices:
            if dev.startswith(device) and 'device' in dev:
                return True
        return False

    @staticmethod
    def get_package_name(app_name: str):
        """
        通过app名称查找packageName
        :return:packageName
        """
        for app_info in AppPackage:
            if app_info["app_name"] == app_name:
                return app_info["package_name"]
        else:
            return None

    def get_device_resolution(self) -> Tuple[int, int]:
        result = self._adb_command("shell wm size")
        if "Override size:" in result:
            resolution = result.split("Override size:")[1].strip().split("x")
            width = int(resolution[0])
            height = int(resolution[1])
            return width, height
        if "Physical size:" in result:
            resolution = result.split("Physical size:")[1].strip().split("x")
            width = int(resolution[0])
            height = int(resolution[1])
            return width, height
        else:
            raise RuntimeError("Failed to get device resolution")

    def call(self, params: Union[str, dict], **kwargs):
        # 验证参数的JSON格式，并将其转换为字典，并执行对应的abd方法
        w_scale_factor = kwargs.get("w_scale_factor", 1)
        h_scale_factor = kwargs.get("h_scale_factor", 1)
        params = self._verify_json_format_args(params)
        action = params["action"]
        # scale_factor = float(ConfigParser.get_config('screenshot', 'scale_factor'))
        # scale_factor = 1 / scale_factor
        if params.get("coordinate"):
            coordinate = []
            for index, point in enumerate(params.get("coordinate")):
                if index == 0:
                    x = round(point * w_scale_factor)
                    coordinate.append(x)
                else:
                    y = round(point * h_scale_factor)
                    coordinate.append(y)
            params["coordinate"] = coordinate

        if params.get("coordinate2"):

            coordinate2 = []
            for index, point in enumerate(params.get("coordinate2")):
                if index == 0:
                    x = round(point * w_scale_factor)
                    coordinate2.append(x)
                else:
                    y = round(point * h_scale_factor)
                    coordinate2.append(y)
            # coordinate2 = [point * scale_factor for point in params["coordinate2"]]
            params["coordinate2"] = coordinate2
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        # 模拟按键事件
        key_event_map = {
            "volume_up": "24",
            "volume_down": "25",
            "power": "26",
            "camera": "27",
            "clear": "28",
        }
        key_code = key_event_map.get(text)
        if key_code is None:
            raise ValueError(f"Unknown key: {text}")
        self._adb_command(f"shell input key_event_map {key_code}")
        return f"Key {text} pressed"

    def _click(self, coordinate: Tuple[int, int]):
        # 模拟点击事件
        x, y = coordinate
        if x >= self.display_width_px - 0:
            x = self.display_width_px - (self.display_width_px * self.out_pull_back_ratio)
        if y >= self.display_height_px - 0:
            y = self.display_height_px - (self.display_height_px * self.out_pull_back_ratio)
        x = round(x)
        y = round(y)
        self._adb_command(f"shell input tap {x} {y}")
        return f"Clicked at ({x}, {y})"

    def _long_press(self, coordinate: Tuple[int, int], time: int = 3):
        # 模拟长按事件
        x, y = coordinate
        if x >= self.display_width_px - 0:
            x = self.display_width_px - (self.display_width_px * self.out_pull_back_ratio)
        if y >= self.display_height_px - 0:
            y = self.display_height_px - (self.display_height_px * self.out_pull_back_ratio)
        x = round(x)
        y = round(y)
        self._adb_command(f"shell input swipe {x} {y} {x} {y} {int(time * 1000)}")
        return f"Long pressed at ({x}, {y}) for {time} seconds"

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int], time: int = 1):
        # 模拟滑动事件
        x1, y1 = coordinate
        x2, y2 = coordinate2

        if x1 >= self.display_width_px - 0:
            x1 = self.display_width_px - (self.display_width_px * self.out_pull_back_ratio)
        if y1 >= self.display_height_px - 0:
            y1 = self.display_height_px - (self.display_height_px * self.out_pull_back_ratio)

        if x2 >= self.display_width_px - 0:
            x2 = self.display_width_px - (self.display_width_px * self.out_pull_back_ratio)
        if y2 >= self.display_height_px - 0:
            y2 = self.display_height_px - (self.display_height_px * self.out_pull_back_ratio)

        # 确保滑动方向正确
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        if y1 > y2:
            self._adb_command(f"shell input swipe {x1} {y1} {x2} {y2} {int(time * 100)}")
            return f"Swiped from ({x1}, {y1}) to ({x2}, {y2})"
        if y2 > y1:
            self._adb_command(f"shell input swipe {x2} {y2} {x1} {y1} {int(time * 100)}")
            return f"Swiped from ({x2}, {y2}) to ({x1}, {y1})"

    def _type(self, text: str):
        # 模拟输入文本事件
        self._adb_command(f"shell am broadcast -a ADB_INPUT_TEXT --es msg {text}")
        return f"Typed: {text}"

    def _system_button(self, button: str):
        # 模拟系统按钮事件
        button_map = {
            "Back": "4",
            "Home": "3",
            "Menu": "82",
            "Enter": "66",
        }
        key_code = button_map.get(button)
        if key_code is None:
            raise ValueError(f"Unknown button: {button}")
        self._adb_command(f"shell input keyevent {key_code}")
        return f"Pressed {button} button"

    def _open(self, text: str):
        # 模拟打开应用事件

        package_name = self.get_package_name(text)
        if package_name is None:
            raise ValueError(f"Unknown app: {text}")
        self._adb_command(f"shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
        return f"Opened app: {text}"

    @staticmethod
    def _wait(time: int = 3):
        tm.sleep(time)
        return f"Waited for {time} seconds"

    def _terminate(self, status: str):
        # 模拟任务终止事件
        return f"Task terminated with status: {status}"

    def _adb_command(self, command: str):
        # 执行ADB命令
        full_command = f"adb -s {self.device} {command}"
        logger.info(f"Executing ADB command: {full_command}")
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ADB command failed: {result.stderr}")
        return result.stdout

    def take_screenshot_and_save(self, screenshot_name="screenshot.png"):
        """
        在设备上截屏并保存到本地目录。
        :param screenshot_name: 截图文件名。
        :return: 本地保存截图的完整路径。
        """
        # 验证输入参数
        if not self.device:
            logger.error("设备对象不能为空")
            return None
        local_directory = ConfigParser.get_config("screenshot", "local_path")
        if not local_directory or not os.path.isdir(local_directory):
            logger.error("本地目录无效")
            return None
        if not screenshot_name:
            logger.error("截图文件名不能为空")
            return None

        # 本地保存截图的路径
        local_path = os.path.join(local_directory, screenshot_name)

        try:
            # 在设备上截屏并保存到/sdcard/screenshot.png
            phone_path = "/sdcard/screenshot.png"
            logger.info("正在通过adb截图...")
            self._adb_command(f"shell screencap -p {phone_path}")
            # 将截图从设备拉取到本地目录
            logger.info(f"将图片保存到{local_path}...")
            self._adb_command(f"pull {phone_path} {local_path}")
            logger.info(f"成功保存到{local_path}")
            resized_path = os.path.join(os.path.dirname(local_path), "screenshot_resized.png")
            h_scale_factor, w_scale_factor = resize_image(local_path, resized_path)
            # 可选：删除设备上的截图以节省空间
            logger.info("正在清除手机上的截图...")
            self._adb_command(f"shell rm {phone_path}")
            return resized_path, h_scale_factor, w_scale_factor
        except Exception as e:
            logger.error(f"出现错误：{e}")
            return None


if __name__ == "__main__":
    m = MobileUse(device="6HJDU19822005857")
    # m.call({"action": "open", "text": "QQ"})
    m.take_screenshot_and_save()
