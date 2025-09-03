#!/usr/bin/env python3
"""
vLLM Performance Monitor for RTX 5000
Real-time monitoring of GPU memory, performance metrics, and optimization suggestions
"""

import time
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path

try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: psutil and GPUtil not available. Install with:")
    print("pip install psutil gputil")

class VLLMMonitor:
    def __init__(self):
        self.monitoring = False
        self.log_file = Path.home() / ".config" / "vllm" / "performance.log"
        self.log_file.parent.mkdir(exist_ok=True)
        
    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                return {
                    'name': data[0],
                    'memory_used': int(data[1]),
                    'memory_total': int(data[2]),
                    'utilization': int(data[3]),
                    'temperature': int(data[4]),
                    'power_draw': float(data[5]) if data[5] != '[N/A]' else 0
                }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
        return None

    def get_system_info(self):
        """Get system CPU and memory information"""
        if not MONITORING_AVAILABLE:
            return {}
            
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

    def check_vllm_processes(self):
        """Check for running vLLM processes"""
        try:
            result = subprocess.run([
                'ps', 'aux'
            ], capture_output=True, text=True)
            
            vllm_processes = []
            for line in result.stdout.split('\n'):
                if 'vllm' in line.lower() and 'python' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        vllm_processes.append({
                            'pid': parts[1],
                            'cpu_percent': parts[2],
                            'memory_percent': parts[3],
                            'command': ' '.join(parts[10:])[:100]
                        })
            return vllm_processes
        except Exception as e:
            print(f"Error checking vLLM processes: {e}")
            return []

    def analyze_performance(self, gpu_info, system_info):
        """Analyze performance and provide recommendations"""
        recommendations = []
        
        if gpu_info:
            memory_usage_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
            
            # Memory recommendations
            if memory_usage_percent > 95:
                recommendations.append("🔴 CRITICAL: GPU memory at {:.1f}%. Risk of OOM. Consider reducing batch size or model size.".format(memory_usage_percent))
            elif memory_usage_percent > 85:
                recommendations.append("🟡 WARNING: GPU memory at {:.1f}%. Monitor closely.".format(memory_usage_percent))
            elif memory_usage_percent < 50:
                recommendations.append("🟢 GPU memory at {:.1f}%. You can likely increase batch size for better throughput.".format(memory_usage_percent))
            
            # Utilization recommendations
            if gpu_info['utilization'] < 50:
                recommendations.append("🟡 Low GPU utilization ({:.1f}%). Check if model is CPU-bound or increase batch size.".format(gpu_info['utilization']))
            elif gpu_info['utilization'] > 95:
                recommendations.append("🟢 High GPU utilization ({:.1f}%). Good performance!".format(gpu_info['utilization']))
                
            # Temperature warnings
            if gpu_info['temperature'] > 80:
                recommendations.append("🔴 High GPU temperature ({}°C). Check cooling.".format(gpu_info['temperature']))
            
        if system_info and 'memory_percent' in system_info:
            if system_info['memory_percent'] > 90:
                recommendations.append("🔴 High system memory usage ({:.1f}%). May cause swapping.".format(system_info['memory_percent']))
                
        return recommendations

    def log_metrics(self, gpu_info, system_info, vllm_processes):
        """Log performance metrics"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'gpu': gpu_info,
            'system': system_info,
            'vllm_processes': len(vllm_processes)
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def display_status(self, gpu_info, system_info, vllm_processes, recommendations):
        """Display current status"""
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print("=" * 80)
        print(f"vLLM Performance Monitor - RTX 5000 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        if gpu_info:
            memory_usage_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
            print(f"🎮 GPU: {gpu_info['name']}")
            print(f"   Memory: {gpu_info['memory_used']:,}MB / {gpu_info['memory_total']:,}MB ({memory_usage_percent:.1f}%)")
            print(f"   Utilization: {gpu_info['utilization']}%")
            print(f"   Temperature: {gpu_info['temperature']}°C")
            if gpu_info['power_draw'] > 0:
                print(f"   Power Draw: {gpu_info['power_draw']:.1f}W")
        else:
            print("❌ GPU information not available")
            
        print()
        
        if system_info:
            print(f"🖥️  System:")
            print(f"   CPU Usage: {system_info['cpu_percent']:.1f}%")
            print(f"   Memory: {system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB ({system_info['memory_percent']:.1f}%)")
        else:
            print("❌ System information not available")
            
        print()
        
        if vllm_processes:
            print(f"🔥 vLLM Processes ({len(vllm_processes)}):")
            for proc in vllm_processes[:3]:  # Show top 3
                print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']}%, MEM {proc['memory_percent']}%")
                print(f"   Command: {proc['command']}")
        else:
            print("⚡ No vLLM processes detected")
            
        print()
        
        if recommendations:
            print("📊 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        else:
            print("✅ Performance looks good!")
            
        print()
        print("Press Ctrl+C to stop monitoring")

    def monitor_loop(self):
        """Main monitoring loop"""
        print("Starting vLLM performance monitor...")
        print("Collecting initial data...")
        
        try:
            while self.monitoring:
                gpu_info = self.get_gpu_info()
                system_info = self.get_system_info()
                vllm_processes = self.check_vllm_processes()
                recommendations = self.analyze_performance(gpu_info, system_info)
                
                self.log_metrics(gpu_info, system_info, vllm_processes)
                self.display_status(gpu_info, system_info, vllm_processes, recommendations)
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
        except Exception as e:
            print(f"\nError in monitoring loop: {e}")

    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

    def show_optimization_tips(self):
        """Show optimization tips for RTX 5000"""
        print("🚀 vLLM Optimization Tips for RTX 5000 (16GB VRAM)")
        print("=" * 60)
        print()
        
        tips = [
            {
                "category": "Memory Management",
                "tips": [
                    "Use gpu_memory_utilization=0.85 (13.6GB for models)",
                    "Start with max_model_len=4096, reduce if OOM occurs",
                    "Enable tensor_parallel_size=1 (single GPU)",
                    "Use block_size=16 for optimal memory allocation"
                ]
            },
            {
                "category": "Model Selection",
                "tips": [
                    "Small models (< 1GB): microsoft/DialoGPT-medium",
                    "Medium models (1-4GB): gpt2-xl, microsoft/DialoGPT-large", 
                    "Large models (4-8GB): Use quantization (AWQ/GPTQ)",
                    "Extra large (> 8GB): Requires aggressive quantization"
                ]
            },
            {
                "category": "Performance Tuning",
                "tips": [
                    "Set max_num_seqs=8 for batch processing",
                    "Enable enable_prefix_caching=true",
                    "Use enable_chunked_prefill=true",
                    "Set max_num_batched_tokens=2048"
                ]
            },
            {
                "category": "Environment Variables",
                "tips": [
                    "export VLLM_ATTENTION_BACKEND=FLASH_ATTN",
                    "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
                    "export OMP_NUM_THREADS=8",
                    "export TOKENIZERS_PARALLELISM=true"
                ]
            }
        ]
        
        for category in tips:
            print(f"📋 {category['category']}")
            for tip in category['tips']:
                print(f"   • {tip}")
            print()

    def generate_report(self):
        """Generate performance report from logs"""
        if not self.log_file.exists():
            print("No log file found. Start monitoring first.")
            return
            
        print("📈 Performance Report")
        print("=" * 40)
        
        try:
            with open(self.log_file, 'r') as f:
                entries = [json.loads(line) for line in f if line.strip()]
                
            if not entries:
                print("No log entries found.")
                return
                
            # Calculate averages for recent entries (last 50)
            recent_entries = entries[-50:]
            
            avg_gpu_util = sum(e['gpu']['utilization'] for e in recent_entries if e['gpu']) / len(recent_entries)
            avg_gpu_memory = sum(e['gpu']['memory_used'] for e in recent_entries if e['gpu']) / len(recent_entries)
            avg_gpu_temp = sum(e['gpu']['temperature'] for e in recent_entries if e['gpu']) / len(recent_entries)
            
            print(f"Recent Performance (last {len(recent_entries)} samples):")
            print(f"  Average GPU Utilization: {avg_gpu_util:.1f}%")
            print(f"  Average GPU Memory Used: {avg_gpu_memory:.0f}MB")
            print(f"  Average GPU Temperature: {avg_gpu_temp:.1f}°C")
            
            # Find peak usage
            peak_memory = max(e['gpu']['memory_used'] for e in entries if e['gpu'])
            peak_util = max(e['gpu']['utilization'] for e in entries if e['gpu'])
            peak_temp = max(e['gpu']['temperature'] for e in entries if e['gpu'])
            
            print(f"\nPeak Usage:")
            print(f"  Peak GPU Memory: {peak_memory:,}MB")
            print(f"  Peak GPU Utilization: {peak_util}%")
            print(f"  Peak Temperature: {peak_temp}°C")
            
        except Exception as e:
            print(f"Error generating report: {e}")

def main():
    monitor = VLLMMonitor()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            monitor.start_monitoring()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop_monitoring()
                
        elif command == "tips":
            monitor.show_optimization_tips()
            
        elif command == "report":
            monitor.generate_report()
            
        elif command == "status":
            # One-time status check
            gpu_info = monitor.get_gpu_info()
            system_info = monitor.get_system_info()
            vllm_processes = monitor.check_vllm_processes()
            recommendations = monitor.analyze_performance(gpu_info, system_info)
            
            monitor.display_status(gpu_info, system_info, vllm_processes, recommendations)
            
        else:
            print("Usage:")
            print("  python vllm_monitor.py monitor    # Start real-time monitoring")
            print("  python vllm_monitor.py status     # One-time status check")
            print("  python vllm_monitor.py tips       # Show optimization tips")
            print("  python vllm_monitor.py report     # Generate performance report")
    else:
        # Default to one-time status
        gpu_info = monitor.get_gpu_info()
        system_info = monitor.get_system_info()
        vllm_processes = monitor.check_vllm_processes()
        recommendations = monitor.analyze_performance(gpu_info, system_info)
        
        monitor.display_status(gpu_info, system_info, vllm_processes, recommendations)

if __name__ == "__main__":
    main()