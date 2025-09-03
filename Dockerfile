# Multi-stage build for Crush
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make gcc musl-dev

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o crush .

# Final stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates git

# Create user
RUN addgroup -g 1000 crush && \
    adduser -D -u 1000 -G crush crush

# Set working directory
WORKDIR /home/crush

# Copy binary from builder
COPY --from=builder /build/crush /usr/local/bin/crush

# Copy configuration templates
COPY --from=builder /build/configs /home/crush/configs

# Switch to non-root user
USER crush

# Set environment variables
ENV CRUSH_CONFIG_PATH=/home/crush/.config/crush
ENV CRUSH_DATA_PATH=/home/crush/.local/share/crush

# Create necessary directories
RUN mkdir -p ${CRUSH_CONFIG_PATH} ${CRUSH_DATA_PATH}

# Entry point
ENTRYPOINT ["crush"]