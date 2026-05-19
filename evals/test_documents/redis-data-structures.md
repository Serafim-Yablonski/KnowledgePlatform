# Redis Data Structures

## Overview

Redis is an in-memory data structure store that supports strings, hashes, lists, sets, sorted sets, bitmaps, hyperloglogs, geospatial indexes, and streams. Its persistence options are RDB snapshots and AOF (Append-Only File). The default Redis port is **6379**.

## Strings

The most basic type. Strings in Redis are binary-safe and can hold up to **512 MB**.

| Command | Description |
|---------|-------------|
| `SET key value [EX seconds]` | Set a string value with optional TTL |
| `GET key` | Retrieve value |
| `INCR key` | Atomically increment integer by 1 |
| `INCRBY key delta` | Atomically increment by delta |
| `SETNX key value` | Set only if key does not exist (used for distributed locks) |
| `GETDEL key` | Get and atomically delete |

```redis
SET counter 0
INCR counter        # returns 1
INCRBY counter 5    # returns 6
```

## Hashes

A Redis hash maps field names to string values. Suitable for representing objects:

```redis
HSET user:1000 name "Alice" age "30" role "admin"
HGET user:1000 name          # "Alice"
HMGET user:1000 name role    # ["Alice", "admin"]
HGETALL user:1000            # all fields and values
```

Hash fields are stored efficiently: hashes with fewer than 128 fields and values under 64 bytes each use a **listpack** encoding (compact array), switching to a hash table beyond these limits. The thresholds are configurable via `hash-max-listpack-entries` and `hash-max-listpack-value`.

## Lists

Redis lists are linked lists of strings. They are ordered by insertion time. Use lists for queues (LPUSH/RPOP), stacks (LPUSH/LPOP), and capped activity logs (LPUSH + LTRIM).

```redis
RPUSH queue "job1" "job2" "job3"
LPOP queue                         # "job1"
BRPOP queue 30                     # blocking pop with 30s timeout
LRANGE queue 0 -1                  # all elements
```

`BLPOP` / `BRPOP` enable efficient worker queues without polling.

## Sets

Unordered collections of unique strings. Operations include union, intersection, and difference:

```redis
SADD tags:post:1 "python" "asyncio" "backend"
SADD tags:post:2 "python" "fastapi"
SINTER tags:post:1 tags:post:2     # {"python"}
SUNIONSTORE all:tags tags:post:1 tags:post:2
SCARD all:tags                     # cardinality
```

Sets use an **intset** encoding for small sets of integers (≤512 values, configurable via `set-max-intset-entries`), making them extremely memory-efficient.

## Sorted Sets

Members are strings, each associated with a floating-point **score**. Members are ordered by score ascending. Sorted sets are ideal for leaderboards, priority queues, and time-series data indexed by timestamp.

```redis
ZADD leaderboard 1500 "alice" 2200 "bob" 900 "charlie"
ZRANK leaderboard "alice"          # rank (0-indexed from lowest)
ZREVRANK leaderboard "bob"         # rank from highest; returns 0
ZRANGEBYSCORE leaderboard 1000 2000  # members with score in range
ZINCRBY leaderboard 300 "alice"    # increment score atomically
```

## Streams

Redis Streams (added in Redis 5.0) are append-only logs with consumer group semantics, similar to Kafka. Each entry is a map of fields assigned an auto-generated or explicit ID:

```redis
XADD events * type "click" user_id "42"
XREAD COUNT 10 STREAMS events 0
XGROUP CREATE events workers $ MKSTREAM
XREADGROUP GROUP workers consumer1 COUNT 1 STREAMS events >
XACK events workers <entry-id>
```

Streams support **consumer groups**: multiple consumers can process entries concurrently, with acknowledgment ensuring each entry is processed once. Unacknowledged entries remain in the **Pending Entries List (PEL)**.

## Expiration and Eviction

Keys can have a TTL set in seconds (`EXPIRE key 3600`) or milliseconds (`PEXPIRE key 3600000`). Expired keys are removed lazily (on access) and actively (background sweep).

When memory reaches `maxmemory`, Redis uses an eviction policy:
- `noeviction` — return errors on write (default)
- `allkeys-lru` — evict least recently used keys
- `volatile-lru` — evict LRU keys with TTL set
- `allkeys-lfu` — evict least frequently used (Redis 4.0+)

## Persistence

**RDB** (Redis Database): point-in-time snapshots. Configured with `save 900 1` (save if 1 change in 900s). Small file, fast restarts, but data between last snapshot and crash is lost.

**AOF** (Append-Only File): logs every write command. `appendfsync everysec` is the recommended setting — at most 1 second of data loss. AOF files grow and must be rewritten periodically (`BGREWRITEAOF`).

**RDB + AOF** together is the highest-durability option.
