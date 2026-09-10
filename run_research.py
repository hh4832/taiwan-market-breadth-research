from market_breadth.config import V8Config
from market_breadth.pipeline import run


if __name__ == "__main__":
    context = run(V8Config())
    print("Completed:")
    for label, path in context["output_paths"].items():
        print(f"  {label}: {path}")
