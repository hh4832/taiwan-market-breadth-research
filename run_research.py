from market_breadth.config import V7Config
from market_breadth.pipeline import run


if __name__ == "__main__":
    context = run(V7Config())
    print("Completed:")
    for label, path in context["output_paths"].items():
        print(f"  {label}: {path}")
