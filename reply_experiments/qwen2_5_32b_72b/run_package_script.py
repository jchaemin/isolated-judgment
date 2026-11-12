"""One package preprocessing script with the Qwen2.5-32B/72B settings. Usage: python run_package_script.py MODEL SCRIPT ARGS... (see run_states.sh)"""
import importlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "preprocessing"), ROOT]
import model_entries as me
model, script = sys.argv[1], sys.argv[2]; sys.argv = sys.argv[2:]
mod = importlib.import_module(script[:-3])
me.use_32b_72b(model, os.environ.get("DATA_32B_72B"))
mod.main()
