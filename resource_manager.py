import psutil
import time
import logging

# Configure logging
logging.basicConfig(filename='/Users/yourusername/logs/resource_manager.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

# List of processes to exclude from management
EXCLUDE_PROCESSES = ['systemd', 'init', 'python3', 'launchd']

def monitor_and_adjust_resources(cpu_threshold=70, ram_threshold=70, sleep_interval=5):
    """
    Monitor CPU and RAM usage and adjust process priorities if usage exceeds the threshold.
    :param cpu_threshold: CPU usage threshold to trigger priority adjustment.
    :param ram_threshold: RAM usage threshold to trigger priority adjustment.
    :param sleep_interval: Time to wait between checks.
    """
    logging.info("Starting resource manager.")
    while True:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] in EXCLUDE_PROCESSES:
                    continue
                
                cpu_usage = proc.info['cpu_percent']
                ram_usage = proc.info['memory_percent']
                
                if cpu_usage > cpu_threshold or ram_usage > ram_threshold:
                    logging.info(f"Process {proc.info['name']} (PID: {proc.info['pid']}) is using {cpu_usage}% CPU and {ram_usage}% RAM.")
                    process = psutil.Process(proc.info['pid'])
                    
                    if cpu_usage > cpu_threshold:
                        process.nice(psutil.IDLE_PRIORITY_CLASS)  # For macOS and Unix systems use psutil.NICE_IDLE
                        logging.info(f"Reduced CPU priority of process {proc.info['name']} (PID: {proc.info['pid']}).")

                    if ram_usage > ram_threshold:
                        process.suspend()  # Suspend process to free memory temporarily
                        logging.info(f"Suspended process {proc.info['name']} (PID: {proc.info['pid']}) due to high RAM usage.")
                        time.sleep(10)  # Wait for 10 seconds before resuming
                        process.resume()
                        logging.info(f"Resumed process {proc.info['name']} (PID: {proc.info['pid']}) after suspension.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logging.warning(f"Failed to manage process {proc.info['name']} (PID: {proc.info['pid']}): {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
        time.sleep(sleep_interval)

if __name__ == "__main__":
    try:
        monitor_and_adjust_resources(cpu_threshold=70, ram_threshold=70, sleep_interval=5)
    except KeyboardInterrupt:
        logging.info("Resource manager stopped.")
