from datetime import datetime
import subprocess

from setuptools import setup


try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str

setup(
    name="dualpipe",
    version="1.0.0" + rev,
    packages=["dualpipe"],
)
