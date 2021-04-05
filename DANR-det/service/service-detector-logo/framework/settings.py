import socket

class BaseConfig:
    APP_VERSION = "v1.0"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # maximum allowed payload to 16 megabytes
    LOG_FILE_NAME = "app_%s.log"%socket.gethostname()
    LOG_LEVEL = "debug"  # 'debug', 'info', 'error', 'warning'
    HOST = "0.0.0.0"
    PORT = 8080
    ENV= "prd"
    CUDA_VISIBLE_DEVICES = ""
    MODEL_PATH = "../models/logo_resneSt50_v1.1.pth"
    THRESHOLD = 0.10
    NUM_CLASSES = 1
    IMG_SIZE = (512,512)


class DevConfig:
    LOG_DIR = "../logs"

class PrdConfig:
    LOG_DIR = "/imagereview/logs"

