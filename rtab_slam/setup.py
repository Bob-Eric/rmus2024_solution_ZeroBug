from distutils.core import setup

setup(
    version="0.0.0",
    scripts=["scripts/odom_publisher.py"],
    packages=["rtab_slam"],
    package_dir={"": "scripts"},
)
