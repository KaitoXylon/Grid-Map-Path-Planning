import time
import numpy as np

def benchmark_run(algorithm, grid, start, goal):
    runtimes = []
    lengths = []
    successes = 0
    result = None

    for _ in range(5):
        start_time = time.time()
        try:
            result = algorithm(grid, start, goal)
            duration = (time.time() - start_time) * 1000
            path_length = len(result["path"])
            if path_length > 0:
                successes += 1
            runtimes.append(duration)
            lengths.append(path_length)
        except Exception as e:
            runtimes.append(float('inf'))
            lengths.append(0)

    return {
        "avg_time": np.mean(runtimes),
        "avg_length": np.mean(lengths),
        "success_rate": (successes / 5) * 100,
        "path": result["path"],
        "explored": result["explored"]
    }
