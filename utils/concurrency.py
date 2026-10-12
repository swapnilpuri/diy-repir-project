from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import MAX_CONCURRENCY


def run_concurrently(fn, items, max_workers=MAX_CONCURRENCY, label=lambda item: ""):
    """
    Apply fn to each item in parallel (useful for slow, I/O-bound LLM calls),
    printing progress as each finishes. Results are returned in input order.
    """
    # items must be sized and iterable. Calls share a process, so fn and any
    # shared client it uses must tolerate execution on multiple threads.
    total = len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit every job; max_workers limits how many run at once.
        futures = {executor.submit(fn, item): item for item in items}
        # Progress follows completion order. Exceptions propagate via result();
        # executor cleanup still waits for submitted jobs before leaving the block.
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()  # surface failures as soon as they happen
            print(f"  [{done}/{total}] {label(futures[future])}")
        # Dict insertion order preserves submission order, keeping categories
        # aligned with their inputs even when slower calls finish last.
        return [future.result() for future in futures]
