"""One package preprocessing script with the Qwen3.5-9B settings. Usage: python run_package_script.py SCRIPT ARGS... (see run_9b_states.sh)"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "preprocessing"), ROOT]
import model_entries as me
me.use_9b(os.environ.get("DATA_9B"))
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(ROOT, "preprocessing", script), run_name="__main__")
