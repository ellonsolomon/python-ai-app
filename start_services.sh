#!/bin/bash  
# start_services.sh  
  
# Start the backend service in the background  
python ./iris_flask.py &  
  
# Start the frontend service in the foreground (this should be the last line)  
python ./iris_frontend.py