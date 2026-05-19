# Python AsyncIO Guide

## Overview

Python's `asyncio` library provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources, running network clients and servers, and other related primitives.

The core concept is the **event loop** — a programming construct that waits for and dispatches events or messages in a program. In asyncio, the event loop runs coroutines, which are functions defined with `async def`.

## The Event Loop

The event loop is the central execution mechanism in asyncio. It manages all async tasks and I/O operations. In Python 3.10+, `asyncio.run()` is the recommended entry point:

```python
import asyncio

async def main():
    print("Hello from asyncio")
    await asyncio.sleep(1)

asyncio.run(main())
```

`asyncio.run()` creates a new event loop, runs the coroutine until completion, and then closes the loop. You should call it only once per program — nested calls raise a `RuntimeError`.

## Coroutines and Tasks

A **coroutine** is a function defined with `async def`. Calling a coroutine function returns a coroutine object, which does not execute until awaited:

```python
async def fetch_data(url: str) -> str:
    await asyncio.sleep(0.1)  # simulates I/O
    return f"data from {url}"
```

A **Task** wraps a coroutine and schedules it to run on the event loop. Create tasks with `asyncio.create_task()`:

```python
async def main():
    task1 = asyncio.create_task(fetch_data("http://api1.example.com"))
    task2 = asyncio.create_task(fetch_data("http://api2.example.com"))
    result1, result2 = await asyncio.gather(task1, task2)
```

`asyncio.gather()` runs multiple awaitables concurrently and returns their results in order.

## asyncio.gather vs asyncio.wait

`asyncio.gather(*coros)` collects results in order and raises the first exception encountered (unless `return_exceptions=True`).

`asyncio.wait(tasks, return_when=...)` gives more control: you can stop when the first task completes (`FIRST_COMPLETED`), when the first exception occurs (`FIRST_EXCEPTION`), or when all complete (`ALL_COMPLETED`).

## Synchronization Primitives

AsyncIO provides async-aware versions of threading synchronization primitives:

- **`asyncio.Lock`** — mutual exclusion; only one coroutine holds the lock at a time
- **`asyncio.Event`** — signal between coroutines; `set()` wakes all waiters
- **`asyncio.Semaphore`** — limits concurrent access to a resource
- **`asyncio.Queue`** — producer/consumer coordination

```python
sem = asyncio.Semaphore(10)

async def limited_fetch(url: str) -> str:
    async with sem:
        return await fetch(url)
```

## The Global Interpreter Lock (GIL) and asyncio

AsyncIO does **not** bypass the GIL. It is single-threaded — only one Python instruction runs at a time. AsyncIO achieves concurrency by switching between coroutines at `await` points (cooperative multitasking), not by running code in parallel.

For true CPU parallelism, use `asyncio.to_thread()` or `concurrent.futures.ProcessPoolExecutor`. The default thread executor in asyncio has a max of `min(32, os.cpu_count() + 4)` threads.

## Streams: asyncio.open_connection

For network I/O, `asyncio.open_connection(host, port)` returns a `(StreamReader, StreamWriter)` pair:

```python
async def fetch_raw(host: str, port: int) -> bytes:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(b"GET / HTTP/1.0\r\n\r\n")
    await writer.drain()
    data = await reader.read(100)
    writer.close()
    await writer.wait_closed()
    return data
```

## Timeouts

Use `asyncio.wait_for(coro, timeout=seconds)` to impose a deadline. On timeout, it raises `asyncio.TimeoutError` and cancels the underlying task:

```python
try:
    result = await asyncio.wait_for(fetch_data(url), timeout=5.0)
except asyncio.TimeoutError:
    print("Request timed out")
```

In Python 3.11+, `asyncio.timeout(seconds)` provides a context manager approach with better task group integration.

## Task Groups (Python 3.11+)

`asyncio.TaskGroup` is the recommended way to run multiple tasks when any failure should cancel siblings:

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(coro1())
        task2 = tg.create_task(coro2())
    # Both tasks complete or both are cancelled on first exception
```

Unlike `gather`, `TaskGroup` cancels all remaining tasks when one raises, and collects all exceptions into an `ExceptionGroup`.

## Common Pitfalls

**Blocking the event loop**: Never call blocking I/O (file reads, `requests.get`, `time.sleep`) directly in a coroutine. Use `asyncio.to_thread()` for blocking calls:

```python
content = await asyncio.to_thread(pathlib.Path("file.txt").read_text)
```

**Forgetting to await**: `result = fetch_data(url)` returns a coroutine object, not the data. Always `await` coroutine calls or wrap in `create_task`.

**Sharing mutable state**: Coroutines run on a single thread, so data races are impossible, but logical races (multiple coroutines modifying a dict between awaits) still occur.
