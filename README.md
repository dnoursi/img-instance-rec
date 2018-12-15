# img-instance-rec
This software package contains code for recognition of image instances through feature descriptors and scalable nearest neighbors (BBF k-d tree).

The API can be called through a single function, ```detect_instances()```, within instance_detection.py. This function takes two arguments, a scene image in which to detect object instances, and a list of object instance images.

Run a demonstration with the following shell command:

> $ python3 instance_detection.py

In this demonstration, an instance of the image of a coke bottle is recognized within a large scene containing furniture, humans, food, and drinks. The demonstration will display the detected image instances, as well as the full scene image.
