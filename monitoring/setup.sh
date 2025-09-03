#!/bin/bash

# Crush + vLLM Monitoring Setup Script
set -e

echo "🚀 Setting up Crush + vLLM Monitoring Stack..."

# Check dependencies
echo "📋 Checking dependencies..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for NVIDIA runtime
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU metrics may not work."
    echo "   Install nvidia-docker2 package if you have NVIDIA GPUs."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{prometheus,grafana,alertmanager}
chmod 755 data/prometheus data/grafana data/alertmanager

# Set proper permissions for Grafana
sudo chown -R 472:472 data/grafana 2>/dev/null || echo "⚠️  Could not set Grafana permissions. Run as root if needed."

# Pull Docker images
echo "📦 Pulling Docker images..."
docker-compose pull

# Start the monitoring stack
echo "🔄 Starting monitoring stack..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "node-exporter:9100")
for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    if curl -sf http://localhost:$port > /dev/null 2>&1; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is not responding on port $port"
    fi
done

# Display access information
echo ""
echo "🎉 Monitoring stack is ready!"
echo ""
echo "📊 Access URLs:"
echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   Prometheus:        http://localhost:9090"
echo "   Alertmanager:      http://localhost:9093"
echo ""
echo "📈 Default Dashboard:"
echo "   vLLM Performance:  http://localhost:3000/d/vllm-crush-dashboard"
echo ""
echo "⚙️  Configuration files:"
echo "   Prometheus:        ./prometheus.yml"
echo "   Alerts:           ./alerts.yml"
echo "   Alertmanager:     ./alertmanager.yml"
echo ""
echo "🔧 To customize email alerts:"
echo "   1. Edit alertmanager.yml with your SMTP settings"
echo "   2. Replace email addresses with your team's emails"
echo "   3. Run: docker-compose restart alertmanager"
echo ""
echo "📝 Next steps:"
echo "   1. Configure your vLLM instance to expose metrics on port 8000"
echo "   2. Install DCGM exporter on GPU nodes if not using Docker"
echo "   3. Customize alert thresholds in alerts.yml as needed"
echo "   4. Set up your notification channels (email/Slack)"
echo ""
echo "🆘 For help: docker-compose logs <service-name>"