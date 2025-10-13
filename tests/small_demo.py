from src.framework import ArgumentationFramework
from src.hc_semantics import weighted_h_categorizer

A = ["a", "b", "c"]
R = [("a", "b"), ("b", "a"), ("c", "b")]
af = ArgumentationFramework.from_lists(A, R)
w = {"a": 0.8, "b": 0.6, "c": 0.9}
print(weighted_h_categorizer(af, w))
