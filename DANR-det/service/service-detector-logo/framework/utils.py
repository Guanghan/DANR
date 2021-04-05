import os
import logging
import logging.handlers

import requests
import flask

def prepared_response(prediction, request_json):
  """
  :param predictions:
  :param parse_result:
  :param get_violation_result_fn:
  :param dataset_violation_codes: {"default": {"p60": {p602"}}}; {"default": {}} #空代表全部
  :param set: (0:porn, 1:tagging, 2:objd, 3:retrieval)
  :return:
  """
  response = []
  output = {}
  output['data_id'] = request_json['data'][0]['data_id']
  output['violation'] = int(prediction > request_json['violation_threshold'])
  output['violation_data'] = []
  for item in request_json['retrieve_source']:
      violation_data = {}
      violation_data['type'] = 'p60'
      violation_data['type_msg'] = '广告'
      violation_data['type_probability'] = '%.3f'%prediction
      violation_data['hits'] = [{
                'dataset_name': item['dataset_name'],
                'probability': float('%.3f'%prediction),
                'md5': '',
                'sub_type': 'p602'}]   # 602 means logo
      output['violation_data'].append(violation_data)
  
  response.append(output)
  return response



def prepared_empty_response(request_json):
    """
    :param predictions:
    :param parse_result:
    :param get_violation_result_fn:
    :param dataset_violation_codes: {"default": {"p20": {p201"}}}; {"default": {}} #空代表全部
    :param set: (0:porn, 1:tagging, 2:objd, 3:retrieval)
    :return:
    """
    response = []
    output = {}
    output['data_id'] = request_json['data'][0]['data_id']
    output['violation'] = 0
    output['violation_data'] = []
    for item in request_json['retrieve_source']:
        violation_data = {}
        violation_data['type'] = 'p60'
        violation_data['type_msg'] = '广告'
        violation_data['type_probability'] = 0
        violation_data['hits'] = []
        output['violation_data'].append(violation_data)

    response.append(output)
    return response


def prepared_response_v2(prediction, request_json):
    """
    :param predictions:
    :param parse_result:
    :param get_violation_result_fn:
    :param dataset_violation_codes: {"default": {"p20": {p201"}}}; {"default": {}} #空代表全部
    :param set: (0:porn, 1:tagging, 2:objd, 3:retrieval)
    :return:
    """
    response = []
    output = {}
    output['data_id'] = request_json['data'][0]['data_id']
    output['violation'] = int(prediction > request_json['violation_threshold']) if prediction is not None else 0
    output['violation_data'] = []
    for item in request_json['retrieve_source']:
        violation_data = {}
        violation_data['type'] = 'p60'
        violation_data['type_msg'] = '广告'
        violation_data['type_probability'] = '%.3f' % prediction if prediction is not None else 0
        violation_data['hits'] = []
        for subtype in item['violation_type']['p60']:
            hit = {
                'dataset_name': item['dataset_name'],
                'probability': None if subtype != 'p602' else '%.3f' % prediction,
                'md5': '',
                'sub_type': subtype}
            violation_data['hits'].append(hit)
        output['violation_data'].append(violation_data)

    response.append(output)
    return response


def checkValidImage(pil_image, min_h=50, min_w=50):
  """
  check if resolution is too small
  :param image: IPLImage
  :return: Boolean
  """
  image_w, image_h = pil_image.size
  return image_w > min_w and image_h > min_h

def parse_request(request: flask.request, threshold: float):
  """
  This function parse request in a complete json format, and generate error
      message and error code.
  :return json: request json format
          json: status
  """
  if not request.is_json:
    return None, {'msg': 'Only support json format', 'code': 400}

  request_json = request.get_json()
  if 'request_no' not in request_json or request_json['request_no'] is None:
      return request_json, {'msg': 'missing request_no', 'code': 400}

  if not isinstance(request_json['request_no'], str):
      return request_json, {'msg': 'request_no is not string', 'code': 400}

  if 'data' not in request_json:
      return request_json, {'msg': 'missing data', 'code': 400}
      
  if 'business_id' not in request_json or request_json['business_id'] is None:
      return request_json, {'msg': 'missing business_id', 'code': 400}

  if not isinstance(request_json['business_id'], str):
      return request_json, {'msg': 'business_id is not string', 'code': 400}

  data = request_json['data']
  if not isinstance(data, list):
     return request_json, {'msg': 'data is not list', 'code':400}
  if len(data) == 0:
     return request_json, {'msg': 'data is empty', 'code':400}
  if len(data) > 1:
     return request_json, {'msg': 'multiple data', 'code':400}

  if not isinstance(data[0], dict):
     return request_json, {'msg': 'item in data is not dictionary', 'code':400}
  if 'content' not in data[0] or 'data_id' not in data[0] or \
         not isinstance(data[0]['content'], str) or not isinstance(data[0]['data_id'], str):
     return request_json, {'msg': 'data format error', 'code':400}
  if 'violation_threshold' not in request_json or request_json['violation_threshold'] is None:
     request_json['violation_threshold'] =  threshold

  if 'retrieve_source' not in request_json:
     return request_json, {'msg': 'retrieve_source not given', 'code': 400}

  if not isinstance(request_json['retrieve_source'], list):
      return request_json, {'msg': 'retrieve_source not a list', 'code': 400}

  if len(request_json['retrieve_source']) == 0:
     request_json['retrieve_source'] = [{'dataset_name':'default',
                                         'violation_type':{"p60":['p602']}}]
                                         #'violation_type':{"p60":[]}}]
  else:
     filter_retrieve_source = []
     for item in request_json['retrieve_source']:
       if 'dataset_name' not in item or "violation_type" not in item:
         return request_json, {'msg': 'retrieve_source format error', 'code':400}

       if not isinstance(item['dataset_name'], str):
         return request_json, {'msg': 'dataset_name is not of type string', 'code': 400}

       if item['dataset_name'] not in ['default', 'fengkong']:
         return request_json, {'msg': 'dataset_name is not default', 'code': 400}

       if not isinstance(item['violation_type'], dict):
         return request_json, {'msg': 'violation_type format error (not dict)', 'code': 400}

       if 'p60' in item['violation_type']:
           for cur_type in item['violation_type']['p60']:
               if cur_type not in ["p601", "p602", "p603", "p604", "p605"]:
                   return request_json, {'msg': 'violation_subtype does not match violation_type', 'code': 400}

       if 'p60' in item['violation_type']:
           if item['violation_type']['p60'] == []:  item['violation_type']['p60'] = ['p602']
           filter_retrieve_source.append(item)

       if  len(item['violation_type']) == 0:
           filter_retrieve_source.append(item)

     if len(filter_retrieve_source) == 0:
       return request_json, {'msg': 'p60 not in retrieve_source', 'code': 400}
     request_json['retrieve_source'] = filter_retrieve_source
  
  return request_json, {'msg': '', 'code':200}


#def parse_request(request):
#  """
#  :param request:
#  :return Boolean: status
#          json dict: request json format
#  """
#  if not request.is_json:
#    return False, {'msg':'Only support json format',
#                   'status': 400}
#
#  request_json = request.get_json()
#  
#  return True, request_json


def get_response(request_json, status, duration, response=[],  model_version=""):
  message_results = {
    "request_no": request_json["request_no"],
    "business_id": request_json["business_id"],
    "service_type": "image",
    "model_version": model_version,
    "status":status['code'],
    "error_message":status['msg'],
    "response": response,
    #"processing times(s)": "%.3f"%duration,
  }
  return message_results


#def log_speaker_configurer(queue, app):
#  log_level_d = {
#    "debug": logging.DEBUG,
#    "warning": logging.WARNING,
#    "error": logging.ERROR,
#    "info": logging.INFO,
#  }
#  _log_level = app.config["LOG_LEVEL"]
#  log_level = log_level_d[_log_level]
#  h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
#  logger = logging.getLogger()
#  logger.addHandler(h)
#  logger.setLevel(log_level)
#  return logger


def log_speaker_configurer(app):
  log_level_d = {
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "info": logging.INFO,
  }
  _log_level = app.config["LOG_LEVEL"]
  log_level = log_level_d[_log_level]
  logger = logging.getLogger()
  logger.setLevel(log_level)
  return logger



def log_listener_configurer(app):
  log_dir = app.config["LOG_DIR"]
  os.makedirs(log_dir, exist_ok=True)
  log_file_name = app.config["LOG_FILE_NAME"]
  log_file = os.path.join(log_dir, log_file_name)
  file_handler = logging.handlers.TimedRotatingFileHandler(
    log_file, when="midnight", interval=1
  )
  file_handler.suffix = "%Y%m%d"
  logging_format = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(funcName)s - %(message)s")
  file_handler.setFormatter(logging_format)
  logger = logging.getLogger()
  logger.addHandler(file_handler)


#This is the listener process top-level loop: wait for logging events
#(LogRecords)on the queue and handle them, quit when you get a None for a
#LogRecord.
def log_listener_process(queue, configurer, app):
    configurer(app)
    while True:
        try:
            record = queue.get(block=True)
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

