# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import logging
import sys
import time
import math
import cv2
import numpy as np
from scipy import ndimage



import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import image_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.client import math_helpers



from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.time_sync import TimedOutError

from PIL import Image as im 
from ultralytics import YOLO
CONFIDENCE_THRESHOLD = 0.5
RED = (0, 0, 255)


# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#model = YOLO('/home/sasuke/Documents/ultralytics-main/jose_code/runs/detect/train4/weights/best.pt') 
model = YOLO("yolo-Weights/yolov8n.pt")

_LOGGER = logging.getLogger(__name__)

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 3  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = '.png'
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = '.jpg'

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
    # print("This is img type: "+str(type(img)))
    
   
    return img, extension


def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')

def find_center(xmin,xmax,ymin,ymax):
   
   
    x = math.fabs(xmax - xmin) / 2.0 + xmin
    y = math.fabs(ymax - ymin) / 2.0 + ymin
    return (x, y)

def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=None, verbose=False):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded.
       Args:
        command_client: robot command client, used to request feedback
        cmd_id: command ID returned by the robot when the trajectory command was sent
        timeout_sec: optional number of seconds after which we'll return no matter what the
                        robot's state is.
        verbose: if we should print state at 10 Hz.
       Return values:
        True if reaches STATUS_AT_GOAL, False otherwise.
    """
    start_time = time.time()

    if timeout_sec is not None:
        end_time = start_time + timeout_sec
        now = time.time()

    while timeout_sec is None or now < end_time:
        feedback_resp = command_client.robot_command_feedback(cmd_id)

        current_state = feedback_resp.feedback.mobility_feedback.se2_trajectory_feedback.status

        if verbose:
            current_state_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(current_state)

            current_time = time.time()
            print('Walking: ({time:.1f} sec): {state}'.format(
                time=current_time - start_time, state=current_state_str),
                end= '                \r')

        if current_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
            return True

        time.sleep(0.1)
        now = time.time()

    if verbose:
        print('block_for_trajectory_cmd: timeout exceeded.')

    return False

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help='Time [ms] to wait before the next capture',
                        type=int, default=100)
    parser.add_argument('-r', '--resize-ratio', help='Fraction to resize the image', type=float,
                        default=1)
    parser.add_argument(
        '--disable-full-screen',
        help='A single image source gets displayed full screen by default. This flag disables that.',
        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
   # bosdyn.client.util.authenticate(robot)
    #autologin
    robot.authenticate("user","bigbubbabigbubba")
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
   
    robot.logger.info('Robot standing.')
    
    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(options.verbose)

    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    robot.logger.info('Commanding robot to stand...')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    


    
    # powered at any point.
  

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
    robot.logger.info('Commanding robot to stand...')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
   # blocking_stand(command_client, timeout_sec=10)
    robot.logger.info('Robot standing.')
    



    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent,
                            resize_ratio=options.resize_ratio) for source in options.image_sources
    ]

    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    keystroke = None
    timeout_count_before_reset = 0
    t1 = time.time()
    image_count = 0
    #grapsing lease
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')
  
    while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
        try:
            images_future = image_client.get_image_async(requests, timeout=0.5)
            while not images_future.done():
                keystroke = cv2.waitKey(25)
                print(keystroke)
                if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                    sys.exit(1)
            images = images_future.result()
        
        except TimedOutError as time_err:
            if timeout_count_before_reset == 5:
                # To attempt to handle bad comms and continue the live image stream, try recreating the
                # image client after having an RPC timeout 5 times.
                _LOGGER.info('Resetting image client after 5+ timeout errors.')
                image_client = reset_image_client(robot)
                timeout_count_before_reset = 0
            else:
                timeout_count_before_reset += 1
        except Exception as err:
            _LOGGER.warning(err)
            continue
            
        for i in range(len(images)):
            
            image, _ = image_to_opencv(images[i], options.auto_rotate)
        
           
            path = im.fromarray(image) 
            detections = model(path)
            boxes = detections[0].boxes

            #for data in detections[0].boxes.data.tolist():
            for box in boxes:
                    # extract the confidence (i.e., probability) associated with the detection
                    confidence = math.ceil((box.conf[0]*100))/100
                    #confidence = data[4]
                    print("Confidence --->",confidence)

                    # filter out weak detections by ensuring the 
                    # confidence is greater than the minimum confidence
                    if float(confidence) < CONFIDENCE_THRESHOLD:
                        continue
                    cls = int(box.cls[i])
                    print("Class name -->", classNames[cls])
                    print(i)
                    
       			#use data name           
                    # if the confidence is greater than the minimum confidence,
                    # draw the bounding box on the frame
                   # xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    xmin, ymin, xmax, ymax = box.xyxy[0]
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax) # convert to int values

                    cv2.rectangle(image, (xmin, ymin) , (xmax, ymax), RED, 2)
                    org = [xmin, ymin]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(image, classNames[cls], org, font, fontScale, color, thickness)
                   
                   # print("if xmin"+str(xmin)+"if xmax"+str(xmax)+"if ymin"+str(ymin)+"if ymax"+str(ymax))
                   # print("The Center"+str(find_center(xmin,xmax,ymin,ymax)))
                    
                    #grab center 
                    center_x,center_y = find_center(xmin,xmax,ymin,ymax)
                    print("CenterX: "+str(center_x))
                    print("CenterY: "+str(center_y))
                
                    # Input data:
                    #   model name
                    #   minimum confidence (between 0 and 1)
                    #   if we should automatically rotate the image
                    pick_vec = geometry_pb2.Vec2(x=center_x, y=center_y)
                    
                    
                    # Take a picture with a camera
                    print("The options: " +str(options.image_sources))
                    robot.logger.info('Getting an image from: %s',options.image_sources)
                    image_responses = image_client.get_image_from_sources(['hand_color_image'])
                    
                    if len(image_responses) != 1:
                        print(f'Got invalid number of images: {len(image_responses)}')
                        print(image_responses)
                        assert False

                    image_grasp = image_responses[0]

                    
                    grasp = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=pick_vec, transforms_snapshot_for_camera=image_grasp.shot.transforms_snapshot,
                    frame_name_image_sensor=image_grasp.shot.frame_name_image_sensor,
                    camera_model=image_grasp.source.pinhole)

                    
                    
                    # Ask the robot to pick up the object
                    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

                    # Send the request
                    
                    cmd_response = manipulation_api_client.manipulation_api_command(
                    manipulation_api_request=grasp_request)
                    

                    breakpoint()
                    """
                    robot.logger.info('Getting an image from: %s', hand_color_image)
                    image_responses = image_client.get_image_from_sources([hand_color_image])
                    """

            #breakpoint()

                    
            cv2.imshow(images[i].source.name, image)
        keystroke = cv2.waitKey(options.capture_delay)
        image_count += 1
        print(f'Mean image retrieval rate: {image_count/(time.time() - t1)}Hz')


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
