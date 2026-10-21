"""Download the Roboflow Universe 'mongolian plate' dataset (984 images, bbox, CC BY 4.0).
Requires a (free) Roboflow account API key:  export ROBOFLOW_API_KEY=xxxx
Then:  python scripts/download_roboflow.py  &&  python scripts/prepare_roboflow_data.py
"""
import os, sys
key = os.environ.get("ROBOFLOW_API_KEY")
if not key:
    sys.exit("Set ROBOFLOW_API_KEY (Roboflow -> Settings -> API). Dataset page: https://universe.roboflow.com/computer-vision-m1xzb/mongolian-plate")
from roboflow import Roboflow
rf = Roboflow(api_key=key)
project = rf.workspace("computer-vision-m1xzb").project("mongolian-plate")
ds = project.version(int(os.environ.get("ROBOFLOW_VERSION", "1"))).download("yolov11", location="data/raw/roboflow")
print("downloaded to", ds.location)
