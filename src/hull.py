import numpy as np
from typing import Optional
from scipy.spatial import ConvexHull

def convex_hull(points: np.ndarray) -> Optional[ConvexHull]:
    """compute the convex hull if possible"""
    if points.ndim != 2:
        return None
    n, d = points.shape
    if n <= d:
        return None
    try:
        return ConvexHull(points)
    except Exception:
        return None
