# Database Transactions and ACID

## ACID Properties

A **transaction** is a sequence of database operations treated as a single unit. Transactions must satisfy four properties, collectively called **ACID**:

### Atomicity

All operations in a transaction either commit together or all roll back. There is no partial success. If a system crash occurs mid-transaction, the database recovers to the state before the transaction began.

Implementation: write-ahead logging (WAL). Changes are written to the WAL before being applied to the data pages. On recovery, the database replays committed transactions and discards uncommitted ones.

### Consistency

A transaction brings the database from one valid state to another valid state. "Valid" is defined by constraints, foreign keys, triggers, and application-level invariants. A transaction that violates a `NOT NULL` or `UNIQUE` constraint is rolled back.

Consistency is partially enforced by the database (schema constraints) and partially the application's responsibility (business rule invariants).

### Isolation

Concurrent transactions execute as if they were serial. The degree of isolation is configurable via **isolation levels**. Higher isolation reduces concurrency anomalies but increases locking overhead.

The SQL standard defines four anomalies:
- **Dirty read**: reading uncommitted data from another transaction
- **Non-repeatable read**: reading the same row twice and getting different values because another transaction committed in between
- **Phantom read**: a re-executed range query returns different rows because another transaction inserted/deleted rows

### Durability

Once a transaction commits, its changes persist even if the system crashes immediately after. This requires flushing the WAL to durable storage (disk sync) before returning success to the client. Setting `synchronous_commit = off` in PostgreSQL trades durability for performance — up to one WAL flush interval of data can be lost.

## Isolation Levels in PostgreSQL

PostgreSQL implements the four SQL standard levels, but its implementation differs subtly:

| Level | Dirty Read | Non-repeatable Read | Phantom Read | Notes |
|-------|------------|---------------------|--------------|-------|
| Read Uncommitted | Not possible | Possible | Possible | Treated as Read Committed |
| **Read Committed** | Not possible | Possible | Possible | **Default** |
| Repeatable Read | Not possible | Not possible | Not possible | Uses snapshot isolation |
| Serializable | Not possible | Not possible | Not possible | Uses SSI |

PostgreSQL's Repeatable Read prevents phantom reads too — a stronger guarantee than the SQL standard requires.

## Read Committed Behavior

In Read Committed, each statement gets a fresh snapshot of committed data. Consider:

```sql
-- Session A
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- (not committed yet)

-- Session B
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- sees OLD balance
COMMIT;

-- Session A
COMMIT;

-- Session B (new query after A commits)
SELECT balance FROM accounts WHERE id = 1;  -- sees NEW balance
```

This is the most common source of lost update bugs in web applications.

## Optimistic vs. Pessimistic Locking

**Pessimistic locking** acquires locks upfront using `SELECT ... FOR UPDATE`. The lock is held until the transaction commits, preventing concurrent modifications:

```sql
BEGIN;
SELECT * FROM inventory WHERE product_id = 42 FOR UPDATE;
UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 42;
COMMIT;
```

**Optimistic locking** does not lock; instead, it records a version number or timestamp at read time and verifies it has not changed at write time:

```sql
UPDATE products SET name = 'New Name', version = version + 1
WHERE id = 42 AND version = 7;
-- If 0 rows updated, another transaction modified the row first
```

Optimistic locking is better for read-heavy workloads; pessimistic for write-heavy or high-contention scenarios.

## Savepoints

A savepoint marks a point within a transaction to which you can roll back without aborting the entire transaction:

```sql
BEGIN;
INSERT INTO orders (id, total) VALUES (1, 100);
SAVEPOINT before_items;
INSERT INTO order_items (order_id, sku) VALUES (1, 'INVALID');  -- might fail
ROLLBACK TO SAVEPOINT before_items;
-- The order INSERT is preserved
COMMIT;
```

Savepoints are used in application frameworks (Django, SQLAlchemy) to implement nested transactions.

## Two-Phase Commit (2PC)

Two-phase commit coordinates transactions across multiple resource managers (multiple databases, message brokers):

1. **Prepare phase**: the coordinator asks all participants to prepare; each writes to its WAL and confirms readiness
2. **Commit phase**: if all prepared, the coordinator sends commit; if any refused, it sends rollback

PostgreSQL supports 2PC via `PREPARE TRANSACTION` and `COMMIT PREPARED`. It is rarely used directly; application frameworks like Spring Boot handle it via XA transactions.

## Advisory Locks

PostgreSQL provides application-level advisory locks that do not lock any database object — they are purely cooperative, governed by your application code:

```sql
SELECT pg_try_advisory_lock(42);  -- returns true if acquired
SELECT pg_advisory_unlock(42);
```

Advisory locks are useful for distributed critical sections (e.g., ensuring only one worker processes a given task at a time) without creating contention on real database rows.
