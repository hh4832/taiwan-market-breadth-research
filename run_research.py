from market_breadth.pipeline import run


if __name__ == "__main__":
    context = run()
    print("Completed:")
    for label, path in context["output_paths"].items():
        print(f"  {label}: {path}")
