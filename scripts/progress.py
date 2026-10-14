import pathlib, sys

run = sys.argv[1] if len(sys.argv) > 1 else "full-benchmark-v1"
pred = pathlib.Path(__file__).resolve().parent.parent / "results" / run / "predictions" if False else None
import pathlib
pred = pathlib.Path(__file__).resolve().parent.parent / "results" / run / "predictions"

arms = [("lexical", 0.001), ("clef-flash", 0.265), ("qwen38-judge", 0.743),
        ("cascade", 0.436), ("clef-27b", 0.835)]

def n(f):
    return sum(1 for _ in open(f)) if f := (arm := pathlib.Path(__file__).parent, pathlib.Path(str(pred) + "/" + f)) and pathlib.Path(str(pred) + "/" + f).exists() else 0
