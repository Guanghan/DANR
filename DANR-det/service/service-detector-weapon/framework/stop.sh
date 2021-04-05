#!/bin/bash
echo "end the wing app"

kill -9 `ps aux |grep gunicorn |grep app:app | awk '{ print $2 }'`  # will kill all of the workers
