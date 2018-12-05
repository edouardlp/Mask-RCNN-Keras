FROM tensorflow/tensorflow:1.12.0-py3
COPY examples/coco/requirements.txt ./
RUN pip install Cython
RUN pip install numpy
RUN pip install --no-cache-dir -r requirements.txt
COPY . maskrcnn
RUN pip install -e ./maskrcnn
WORKDIR /usr/src/app
COPY examples/coco/* ./
CMD [ "python", "./eval_task.py" ]