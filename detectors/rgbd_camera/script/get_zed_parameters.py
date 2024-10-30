# Script created for getting parameters of the Zed camera.
# Use this to update the launch file.

import pyzed.sl as sl

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
# Change the parameters here!!!!!!!!!!!!
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.sdk_verbose = 0


err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED Camera.", err)
    exit(1)
elif zed.is_opened():
  print("Camera available!")
# else:
#   # Open the camera
# err = zed.open(init_params)
# if err != sl.ERROR_CODE.SUCCESS:
#     print("Failed to open ZED Camera.", err)
#     exit(1)
# elif zed.is_opened():
#   print("Camera available!")

# try:
#   zed.is_opened()
#   # zed = sl.Camera()
#   print("Camera available!")
# except:
#   # Open the camera
#   err = zed.open(init_params)
#   if err != sl.ERROR_CODE.SUCCESS:
#       exit(1)


# Get camera information (ZED serial number)
zed_serial = zed.get_camera_information().serial_number
camera_info = zed.get_camera_information()
print("Zed camera serial number: {0}".format(zed_serial))
# print("Camera resolution WVGA:", camera_info.camera_configuration.resolution)
print("Camera FPS:", camera_info.camera_configuration.fps)
print("Camera calibration parameters:", )

calibration_parameters = camera_info.camera_configuration.calibration_parameters
# Extract focal lengths and principal point
fx = calibration_parameters.left_cam.fx
fy = calibration_parameters.left_cam.fy
cx = calibration_parameters.left_cam.cx
cy = calibration_parameters.left_cam.cy
# Access distortion coefficients (example: k1, k2)
k1 = calibration_parameters.left_cam.disto[0]
k2 = calibration_parameters.left_cam.disto[1]
print("Focal Length (x):", fx)
print("Focal Length (y):", fy)
print("Principal Point (x):", cx)
print("Principal Point (y):", cy)
# print("distortion coefficients:",k1,k2)
# Close the camera
zed.close()