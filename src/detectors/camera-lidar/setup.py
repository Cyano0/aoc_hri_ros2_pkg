from setuptools import setup

package_name = 'human_camlidar_det_uol'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'human_camlidar_det_uol.marker_to_lidar',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/marker_to_lidar.launch.py']),
    ],
    install_requires=['setuptools', 'human_tracker_uol'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Description of the package',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'marker_to_lidar = human_camlidar_det_uol.marker_to_lidar:main',
        ],
    },
)