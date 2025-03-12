import configparser
import os


class ConfigParser:
    """
    读取配置文件信息
    """
    config_dic = {}

    @classmethod
    def get_config(cls, sector, item):
        value = None
        try:
            value = cls.config_dic[sector][item]
        except KeyError:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.conf')
            cf = configparser.ConfigParser()
            cf.read(config_path, encoding='utf8')
            value = cf.get(sector, item)
            cls.config_dic[sector][item] = value
        finally:
            return value

    def set_config(self, sector, item, value):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.conf')
        cf = configparser.ConfigParser()
        cf.read(config_path, encoding='utf8')
        cf.set(sector, item, value)
        cf.write(open(config_path, 'w', encoding='utf8'))