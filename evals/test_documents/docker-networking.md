# Docker Networking

## Network Drivers

Docker supports multiple network drivers, each suited to different use cases:

- **bridge** — the default for standalone containers. Creates a virtual network on the host with a software bridge. Containers on the same bridge network can communicate; containers on different bridges are isolated.
- **host** — removes network isolation and binds the container directly to the host's network. The container shares the host's IP and port space. Available on Linux only.
- **overlay** — spans multiple Docker hosts (Docker Swarm). Uses VXLAN encapsulation; requires a key-value store.
- **macvlan** — assigns a MAC address to each container, making it appear as a physical device on the network. Used for legacy apps that expect direct layer-2 access.
- **none** — disables all networking; the container has only the loopback interface.

## Default Bridge Network vs. User-Defined Bridges

Containers connected to the **default bridge** (`docker0`) can communicate by IP only — DNS-based discovery is not available. This is a legacy default kept for backward compatibility.

**User-defined bridge networks** are strongly preferred:
- Automatic DNS resolution by container name
- Better isolation: only containers explicitly connected to the network can communicate
- Containers can be attached/detached without restart

```bash
docker network create my-app-net
docker run --network my-app-net --name api my-api-image
docker run --network my-app-net --name db postgres:16
# "db" is resolvable by name inside "api"
```

## Port Publishing

`-p host_port:container_port` binds a container port to the host. Docker inserts iptables rules to forward traffic. Publishing to `0.0.0.0` (default) exposes the port on all host interfaces; use `127.0.0.1:5432:5432` to bind only to localhost:

```bash
docker run -p 127.0.0.1:5432:5432 postgres:16
```

Without `-p`, a container port is accessible only from containers on the same network — not from the host.

## Docker Compose Networking

Docker Compose creates a dedicated bridge network per project (named `<project>_default`). All services in the same `docker-compose.yml` join this network automatically and are resolvable by service name:

```yaml
services:
  api:
    image: my-api
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: mydb
```

Inside `api`, `db:5432` resolves correctly. Override the default network or add additional networks in the `networks:` block.

## DNS Resolution

Docker uses an embedded DNS server at `127.0.0.11` (within containers). Container names and service names are registered automatically on user-defined networks. The DNS server also handles load balancing for scaled services by returning all IPs for a service name.

Containers on the default bridge network use the host's `/etc/resolv.conf` and cannot use container-name DNS.

## Network Policies and Isolation

Docker does not have Kubernetes-style NetworkPolicy objects. Isolation is achieved by:
1. Placing services on separate user-defined networks
2. Only connecting containers that need to communicate to the same network
3. Not publishing ports unnecessarily

A container can be attached to multiple networks simultaneously:

```bash
docker network connect monitoring-net api-container
```

## MTU and Performance

The default MTU for bridge networks is 1500 bytes. In cloud environments where the host's MTU is lower (e.g., 1450 bytes for AWS with VXLAN), you must configure Docker's MTU to avoid silent packet fragmentation:

```json
// /etc/docker/daemon.json
{
  "mtu": 1450
}
```

Overlay networks add a VXLAN header (50 bytes) and require a further reduced MTU of 1450 on the underlying network.

## IPv6

Docker supports IPv6 on user-defined networks. Enable it in `daemon.json`:

```json
{
  "ipv6": true,
  "fixed-cidr-v6": "fd00::/80"
}
```

Each container receives a unique IPv6 address derived from its MAC address.

## Troubleshooting

Common tools for debugging Docker network issues:

- `docker network inspect <name>` — shows containers, IP ranges, and driver options
- `docker exec <container> ping <target>` — tests connectivity
- `nsenter --target $(docker inspect -f '{{.State.Pid}}' <c>) --net ip addr` — view the container's network namespace from the host
- `tcpdump` inside a container: use `--cap-add=NET_ADMIN` when running

Packets dropped silently usually indicate MTU mismatch or iptables rules blocking traffic. Run `iptables -L DOCKER-USER` to inspect custom rules.
