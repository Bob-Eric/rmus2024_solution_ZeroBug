from distutils.core import setup

setup(
    version="0.0.0",
    scripts=["scripts/ekf_transfer.py"],
    packages=["rtab_slam"],
    package_dir={"": "scripts"},
)
