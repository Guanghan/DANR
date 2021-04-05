from time import time
import requests
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from utils import *
from models.detector import CenterNetDetector

# set up default values
DEBUG = False

# set up flask app
app = Flask(__name__)
app.config.from_object("settings.BaseConfig")
if app.config["ENV"] == "dev":
    app.config.from_object("settings.DevConfig")
elif app.config["ENV"] == "prd":
    app.config.from_object("settings.PrdConfig")

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["3000 per minute", "60 per second"],
)


class ImageScanner():

  def __init__(self, **kwargs):
    self.logger = kwargs['logger']
    self.detector = CenterNetDetector(kwargs['config']['MODEL_PATH'], kwargs['config']['NUM_CLASSES'])
    self.img_size = kwargs['config']['IMG_SIZE']
    self.worker_timeout = 8


  def predict_entry_point(self, request):
    start_time = time()

    # Download content
    try:
      url = request["data"][0]["content"]
      req = requests.request('GET', url, headers={'appauthoritykey': 'imagescanner_app', 'appname': 'imagescanner_app'}, timeout=1)
      self.logger.info("download time= %.3f" %(time()-start_time))
    except Exception as e:
      error_msg = "Can't get request, url=%s" % (url)
      self.logger.error(error_msg)
      self.logger.error(str(e))
      return None, {'msg': error_msg, 'code': 400}

    # Open Image
    try:
      image = Image.open(BytesIO(req.content)).convert("RGB")
      if not checkValidImage(image):
        error_msg = "Resolution (%d, %d) is too small for url=%s" % (image.size[0], image.size[1], url)
        return None, {'msg': error_msg, 'code': 400}
    except Exception as e:
      error_msg = "Can't get image, url=%s" % (url)
      self.logger.error(error_msg)
      self.logger.error(str(e))
      return None, {'msg': error_msg, 'code': 400}

    try:
      image = self.detector.preprocess_image(image, self.img_size)
      self.logger.info("preprocess time= %.3f" %(time()-start_time))
      prob = self.detector.inference(image)
      self.logger.info("inference time= %.3f" %(time()-start_time))
    except Exception as e:
      error_msg = "Can't get prob, url=%s" % (url)
      self.logger.error(error_msg)
      self.logger.error(str(e))
      self.request_queue.get(key)
      return None, {'msg': error_msg, 'code': 400}

    self.logger.debug("Total exec time= %.3f" %(time()-start_time))
    return prob, {'msg': '', 'code': 200}


@app.route("/imagereview", methods=["POST"])
@limiter.limit('60/second;3000/minute')
def image_review_api():
  """ for RESTful API
  :return:
  """
  start_time = time()
  request_json, status = parse_request(request, app.config["THRESHOLD"])
  if status['code'] != 200:
      duration = time() - start_time
      return get_response(request_json, status, duration)

  if 'p602' in request_json['retrieve_source'][0]['violation_type']['p60']:
      result, status = imagescanner.predict_entry_point(request_json)
  else:
      response = prepared_empty_response(request_json)
      #response = prepared_response_v2(None, request_json)
      duration = time() - start_time
      #res = get_response(request_json, status, duration, response=[])
      res = get_response(request_json, status, duration, response)
      return res

  if status['code'] != 200 or result is None:
      duration = time() - start_time
      return get_response(request_json, status, duration)

  response = prepared_response(result, request_json)
  #response = prepared_response_v2(result, request_json)
  duration = time() - start_time
  res = get_response(request_json, status, duration, response)
  return res


# mp_logger
log_listener_configurer(app)
app.logger = log_speaker_configurer(app)

# image scanner
imagescanner = ImageScanner(logger=app.logger, config=app.config)

if __name__ == "__main__":
  print("app.name=", app.name, "; version=", app.config["APP_VERSION"])
  app.run(host='0.0.0.0', port=5556, debug=DEBUG, threaded=False)
