# Multi-stage build for the Modal Inference Blueprint operator.
#
# Build:  docker build -t modal-inference-operator .
# Run:    docker run -v ./config:/app/config modal-inference-operator

FROM rust:1.82-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY operator/ operator/

# Build dependencies first (cached layer)
RUN mkdir -p operator/src && echo "fn main() {}" > operator/src/main.rs
RUN cargo build --release --bin standalone 2>/dev/null || true

# Build actual code
COPY . .
RUN cargo build --release --bin standalone

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/standalone /app/standalone
COPY config/example.toml /app/config/operator.toml

EXPOSE 8080 9090
CMD ["/app/standalone"]
