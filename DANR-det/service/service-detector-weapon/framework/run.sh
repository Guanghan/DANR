#!/bin/sh
gunicorn -w 8 -b 0.0.0.0:5556 app:app
