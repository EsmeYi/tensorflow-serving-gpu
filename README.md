# Tensorflow:Serving-GPU
Serve a pre-trained model (Mask-RCNN, Faster-RCNN, SSD) on Tensorflow:Serving.

TensorFlow Serving is designed for deployment ML models inference. For an in-depth overview, please head to [TF-Serving](https://www.tensorflow.org/tfx/guide/serving) document.


## 1. Install [docker](https://docs.docker.com/install/) and [nvidia-docker2]()

For Ubuntu, please follow the Docker official document.

For RHEL, please install the docker distribution (v1.13.1) instead of docker-ce or docker-ee, then install nvidia-container-runtime-hook.

~~~
$ docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
$ yum remove nvidia-docker

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-runtime.repo

$ yum install -y nvidia-container-runtime-hook

$ docker run --rm nvidia/cuda-ppc64le nvidia-smi
~~~


## 2. Install TensorFlow:Serving-GPU with docker

For x86:

	$ docker pull tensorflow/serving:latest-gpu

For ppc64le:

Build from dockerfiles: [tensorflow-serving-ppc64le](https://github.com/ppc64le/build-scripts/tree/master/tensorflow-serving/Dockerfiles)

Note: Multi-stage builds are requiring Docker 17.05 or higher, if you are using docker 1.13.1, you need make a little change on Dockerfiles.

Or pull a built image from docker hub

	$ docker pull ibmcom/tensorflow-serving-ppc64le:latest-gpu

## 3. Prepare a severable model

(1) Download a pre-trained Mask-RCNN model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

(2) Export a model for serving

	$ git clone https://github.com/tensorflow/models.git
 	$ cd research
  
Make a little change in the **write-saved-model()** function of object_detection/exporter.py to make it produce an unfrozen graph.

~~~
def write_saved_model(saved_model_path,
                      trained_checkpoint_prefix,
                      inputs,
                      outputs):
  saver = tf.train.Saver()
  with session.Session() as sess:
    saver.restore(sess, trained_checkpoint_prefix)
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
	......
    builder.save(as_text=True) # True:pbtxt; False:pb
~~~
  
	$ python setup.py install
	$ python setup.py build
	$ export PYTHONPATH=$PYTHONPATH:/path/to/models/research/:/path/to/models/research/slim
	
And then export the inference graph
	
	$ python object_detection/export_inference_graph.py \
	--input_type image_tensor --pipeline_config_path path/to/downloaded_model/pipeline.config \
	--trained_checkpoint_prefix path/to/downloaded_model/model.ckpt \
	--output_directory path/to/saved_model
	
You will see the saved_model folder like this

~~~
--saved_model
  --version
    --1
      saved_model.pbtxt
      --variables
	variables.data-00000-of-oooo1
	variables.index
~~~

Remember the **signature def(including signature name, input name, out keys, etc)** or check them in .pbtxt file, which are useful later.

There is a more convenient approch to show all signature def and tag set:

	$ saved_model_cli show --dir /path/saved_model_dir/ \
	--tag_set serve --signature_def serving_default 

## 4. Launch the model server 

	$ docker run -t --rm --name maskrcnn-server \
	-p 9000:8500 \
	-v "/path/to/saved_model/versions:/models/model_name" \
	-e MODEL_NAME= model_name -t tensorflow-serving-gpu:latest

	
## 5. Generating client requests

| dataset | [VOC-2017](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) |
| :----:  | :----: |
| Size | 300*300 |
| Channel    | 3 |
| Amount | 4952 |

client_requirements.txt

~~~
grpcio==1.12.1
html5lib==0.9999999
numpy==1.14.3
Pillow==5.1.0
protobuf==3.6.1
requests==2.18.4
six==1.12.0
tensorflow-serving-api==1.13.0
~~~

client_request.py

| arguments | command | defualt|
| :----:  | :----: |:----: |
| server url | -s | 172.17.0.1:9000|
| model name    | -mn | mrcnn| 
| signature name| -sn | serving_default |
| input name | -in | inputs |
| output size | -o | 100 |
| VOC root directory | -voc | ./VOCdevkit/VOC2007 |
| batch size | -b | 32 |


## Result

| model | batch size |latency (ms) |
| :----:  | :----: | :----: |
| Mask-RCNN    | 32 | 830306 ms|
| Faster-RCNN   | 32 | 346567 ms|
| SSD   | 5 | 173585 ms|
