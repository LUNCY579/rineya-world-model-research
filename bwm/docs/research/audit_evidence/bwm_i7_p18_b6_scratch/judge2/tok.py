import sys, random, json
ROOT=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from transformers import AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("lmpaths", ROOT + "/bwmn/lm.py")
# avoid importing torch-heavy lm: read MODELS path by text
src=open(ROOT+"/bwmn/lm.py",encoding="utf-8").read()
import re
cores_line=[l for l in src.splitlines() if l.startswith("CORES")]
print(cores_line)
