# img-instance-rec
[https://github.com/dnoursi/img-instance-rec](https://github.com/dnoursi/img-instance-rec)


This software package contains code for recognition of image instances within large scene images. We implement algorithms for  descriptor vector construction, and scalable nearest neighbors with a BBF k-d tree.

The API can be called through a single function, `detect_instances(scene, instances)`, within `instance_detection.py`. This function takes two arguments, a scene image in which to detect object instances, and a list of object instance images.

Run a demonstration with the following shell command:

> $ python3 instance_detection.py

In this demonstration, an instance of the image of a coke bottle is recognized within a large scene containing furniture, humans, food, and drinks. The demonstration will display the detected image instances, as well as the full scene image.

## Code organization

The code implementations of algorithms for detecting interest points, generating features, and executing nearest neighors is within `features.py` and `canny.py`. 

Helper code for handling image I/O with various third party libraries is primarily in `util.py`, `visualize.py`, and `gen_data.py` . 

`instance_detection.py` contains the primary high-level API function `detect_instances(scene, instances)`, as well as code for running the demonstration as described above.
