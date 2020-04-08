FROM tensorflow/tensorflow:latest-py3
ADD train.py /
RUN pip install pymysql
RUN pip install requests
RUN pip install python-dateutil
CMD [ "python", "-u", "/train.py" ]