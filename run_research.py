from market_breadth.config import V9Config
from market_breadth.pipeline_v9 import run_v9


if __name__ == "__main__":
    context = run_v9(V9Config())
    print("Completed:")
    for label, path in context["output_paths"].items():
        print(f"  {label}: {path}")
