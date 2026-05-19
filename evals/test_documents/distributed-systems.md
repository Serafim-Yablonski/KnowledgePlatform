# Distributed Systems Fundamentals

## The CAP Theorem

The CAP theorem states that a distributed system can provide at most **two of three** guarantees simultaneously:

- **Consistency (C)**: every read receives the most recent write or an error
- **Availability (A)**: every request receives a response (not necessarily the most recent data)
- **Partition Tolerance (P)**: the system continues operating despite network partitions

In practice, network partitions are unavoidable, so the real choice is **CP vs. AP**:

- **CP systems** (e.g., HBase, ZooKeeper): refuse requests during a partition to preserve consistency
- **AP systems** (e.g., Cassandra, DynamoDB): serve stale data during a partition to remain available

Relational databases like PostgreSQL are typically **CP** — during a partition, the primary refuses writes or the replica falls behind, but you never read inconsistent data from a single node.

## Consistency Models

CAP's "consistency" is linearizability — a specific strong model. There is a broader spectrum:

### Strong Consistency (Linearizability)

All reads reflect the most recent write. Operations appear instantaneous and globally ordered. Used by etcd and ZooKeeper for coordination. Expensive: requires consensus (Paxos, Raft) on every write.

### Sequential Consistency

All nodes see operations in the same order, but not necessarily real-time order. Weaker than linearizability but allows batching.

### Eventual Consistency

Given no new writes, all replicas will eventually converge. Reads may return stale data. Systems like Cassandra use this model with tunable consistency (quorum reads/writes can approximate strong consistency).

### Read-Your-Writes Consistency

A weaker guarantee: after a write, the same client always reads its own latest write. Implemented with session stickiness or client-side version tracking.

## Consensus Protocols

Consensus is needed when multiple nodes must agree on a value (e.g., leader election, distributed transactions).

**Raft** (used by etcd, CockroachDB, TiKV): nodes elect a leader; all writes go through the leader and are replicated to a majority before committing. If the leader fails, a new one is elected. Raft is designed to be understandable; its log-based approach maps naturally to state machine replication.

**Paxos** (used by Google Spanner, Chubby): more general and complex. Multi-Paxos adds leader election on top of basic Paxos to amortize prepare phases.

Both require a majority quorum (n/2 + 1 nodes) to make progress. With 3 nodes, 1 failure is tolerated; with 5 nodes, 2 failures.

## Replication Strategies

**Primary-replica (leader-follower)**: all writes go to the primary; replicas receive changes asynchronously. Reads can be served from replicas. PostgreSQL streaming replication follows this model. Async replication means a crash can lose committed transactions — synchronous replication (`synchronous_standby_names`) eliminates this at the cost of latency.

**Multi-primary**: multiple nodes accept writes. Conflict resolution is required (last-write-wins, CRDTs, or application-level merging). Used by CockroachDB and Dynamo-style systems.

**Quorum reads/writes**: a write is acknowledged when W nodes confirm; a read is satisfied when R nodes respond. Setting `W + R > N` guarantees reading the latest write in the absence of failures.

## Distributed Transactions

Distributed transactions coordinate writes across multiple services or databases. Two approaches:

**2PC (Two-Phase Commit)**: atomic but blocking — if the coordinator fails during the commit phase, participants are stuck in an uncertain state. Maximum coupling between services.

**Saga pattern**: a sequence of local transactions, each publishing an event that triggers the next step. Failures trigger compensating transactions. No distributed locks; higher availability but eventual consistency only. The Saga is either *choreography-based* (services listen to events) or *orchestration-based* (a central saga orchestrator drives the flow).

## The Fallacies of Distributed Computing

Peter Deutsch listed 8 assumptions that newcomers make:

1. The network is reliable
2. Latency is zero
3. Bandwidth is infinite
4. The network is secure
5. Topology doesn't change
6. There is one administrator
7. Transport cost is zero
8. The network is homogeneous

All eight are false in production. Design for failure: use retries with exponential backoff, circuit breakers, timeouts, and idempotent operations.

## Idempotency and At-Least-Once Delivery

Message queues typically guarantee **at-least-once delivery** — a message may be delivered more than once (e.g., after a consumer crash before acknowledging). Operations must be idempotent: applying the same message twice produces the same result as applying it once.

Common patterns:
- **Idempotency key**: include a unique key in each message; the consumer deduplicates using a set of processed keys
- **Natural idempotency**: design mutations to be naturally idempotent (SET operations instead of INCREMENT)

**Exactly-once** delivery requires cooperation between the message broker and the consumer's database (transactional outbox pattern, Kafka transactions).

## Clock Synchronization

Distributed systems cannot rely on wall clocks being in sync. NTP achieves ~1–10ms accuracy under ideal conditions; GPS-disciplined clocks can reach microseconds (Google TrueTime).

For ordering events without a global clock:
- **Lamport timestamps**: logical clock incremented on each event; preserve causal order
- **Vector clocks**: track causality per process; can detect concurrent events
- **Hybrid logical clocks (HLC)**: combine physical time with Lamport logic; used by CockroachDB
