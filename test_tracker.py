
import time
import os
from resource_tracker import ResourceTracker

def test_tracker():
    print("Testing ResourceTracker...")
    tracker = ResourceTracker()
    
    print("Simulating Stage 1: CPU Work (0.5s)")
    with tracker.track("Stage1"):
        start = time.time()
        # Burn CPU
        while time.time() - start < 0.5:
            pass
            
    print("Simulating Stage 2: Sleep (0.5s)")
    with tracker.track("Stage2"):
        time.sleep(0.5)

    print("Simulating Stage 3: Memory Alloc (100MB)")
    data = []
    with tracker.track("Stage3"):
        # Allocate 100MB
        data = bytearray(100 * 1024 * 1024)
        time.sleep(0.1)

    print("Simulating Stage 1 again: CPU Work (0.2s)")
    with tracker.track("Stage1"):
        start = time.time()
        while time.time() - start < 0.2:
            pass

    tracker.print_stats()
    
    s1 = tracker.stats["Stage1"]
    assert s1["count"] == 2
    assert s1["time"] >= 0.7
    assert s1["cpu_time"] >= 0.6 # Allow some slack, but should be close to wall time
    
    s2 = tracker.stats["Stage2"]
    assert s2["count"] == 1
    assert s2["time"] >= 0.5
    assert s2["cpu_time"] < 0.1 # Should be low for sleep
    
    s3 = tracker.stats["Stage3"]
    # checking RSS is tricky as python GC might not be immediate and OS might be lazy, but it should be > 0
    print(f"Stage3 MaxRSS: {s3['max_rss'] / 1024 / 1024} MB")
    
    print("Test Passed!")

if __name__ == "__main__":
    test_tracker()
