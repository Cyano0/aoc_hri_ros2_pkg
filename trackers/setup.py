from setuptools import setup

package_name = 'human_tracker_uol'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Description of the package',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)