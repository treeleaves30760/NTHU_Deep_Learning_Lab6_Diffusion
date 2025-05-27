#!/usr/bin/env python3
import subprocess
import os
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import argparse
from typing import List, Dict, Any
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('run_experiments.log'),
            logging.StreamHandler()
        ]
    )


class ExperimentRunner:
    """Class to manage multiple training experiments."""

    def __init__(self):
        self.processes = []
        self.experiment_configs = []

    def create_experiment_configs(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create different experiment configurations."""
        configs = []

        # Experiment 1: Baseline with cosine schedule
        config1 = base_config.copy()
        config1.update({
            'device': 'cuda:0',
            'output_dir': './outputs/exp1_cosine_baseline',
            'wandb_run_name': 'exp1_cosine_baseline',
            'schedule_type': 'cosine',
            'base_channels': 128,
            'learning_rate': 1e-4,
            'batch_size': 16,
            'num_timesteps': 1000,
            'sample_freq': 20
        })
        configs.append(config1)

        # Experiment 2: Higher capacity model with linear schedule
        config2 = base_config.copy()
        config2.update({
            'device': 'cuda:1',
            'output_dir': './outputs/exp2_cosine_large',
            'wandb_run_name': 'exp2_cosine_large',
            'schedule_type': 'cosine',
            'base_channels': 256,
            'learning_rate': 2.5e-5,
            'batch_size': 16,
            'num_timesteps': 1000,
            'sample_freq': 20
        })
        configs.append(config2)

        # Experiment 3: Fast training with fewer timesteps
        config3 = base_config.copy()
        config3.update({
            'device': 'cuda:2',
            'output_dir': './outputs/exp3_fast_training',
            'wandb_run_name': 'exp3_fast_training',
            'schedule_type': 'cosine',
            'base_channels': 96,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_timesteps': 500,
            'sample_freq': 20
        })
        configs.append(config3)

        return configs

    def config_to_args(self, config: Dict[str, Any]) -> List[str]:
        """Convert config dictionary to command line arguments."""
        args = ['python', 'train.py']

        for key, value in config.items():
            if key == 'device':
                # Set CUDA_VISIBLE_DEVICES instead of passing device argument
                continue
            args.append(f'--{key}')
            args.append(str(value))

        return args

    def run_experiment(self, config: Dict[str, Any], exp_id: int) -> Dict[str, Any]:
        """Run a single experiment."""
        device = config.get('device', 'cuda:0')
        gpu_id = device.split(':')[-1] if ':' in device else '0'

        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

        # Create output directory
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Convert config to command line arguments
        cmd_args = self.config_to_args(config)

        logging.info(f"Starting experiment {exp_id} on GPU {gpu_id}")
        logging.info(f"Command: {' '.join(cmd_args)}")
        logging.info(f"Output directory: {output_dir}")

        try:
            # Start the training process
            log_file = os.path.join(
                output_dir, f'training_log_exp{exp_id}.txt')

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd_args,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

            self.processes.append(process)

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                logging.info(f"Experiment {exp_id} completed successfully")
                return {'exp_id': exp_id, 'status': 'success', 'return_code': return_code}
            else:
                logging.error(
                    f"Experiment {exp_id} failed with return code {return_code}")
                return {'exp_id': exp_id, 'status': 'failed', 'return_code': return_code}

        except Exception as e:
            logging.error(f"Error running experiment {exp_id}: {str(e)}")
            return {'exp_id': exp_id, 'status': 'error', 'error': str(e)}

    def run_all_experiments(self, configs: List[Dict[str, Any]], max_workers: int = 3):
        """Run all experiments in parallel."""
        self.experiment_configs = configs

        # Check GPU availability
        self.check_gpu_availability()

        logging.info(f"Starting {len(configs)} experiments in parallel")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            futures = []
            for i, config in enumerate(configs):
                future = executor.submit(self.run_experiment, config, i + 1)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error collecting result: {str(e)}")
                    results.append({'status': 'error', 'error': str(e)})

        return results

    def check_gpu_availability(self):
        """Check if required GPUs are available."""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                    capture_output=True, text=True)
            gpu_count = len(result.stdout.strip().split('\n'))

            if gpu_count < 3:
                logging.warning(
                    f"Only {gpu_count} GPUs available, but 3 experiments planned")
                logging.warning("Some experiments may run sequentially")
            else:
                logging.info(
                    f"Found {gpu_count} GPUs, sufficient for parallel training")

        except subprocess.CalledProcessError:
            logging.warning("Could not check GPU availability")

    def cleanup(self):
        """Cleanup running processes."""
        logging.info("Cleaning up running processes...")
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                logging.info(f"Terminating process {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Force killing process {process.pid}")
                    process.kill()


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logging.info("Received interrupt signal, cleaning up...")
    if 'runner' in globals():
        runner.cleanup()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple DDPM training experiments')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing dataset')
    parser.add_argument('--image_dir', type=str, default='../iclevr',
                        help='Directory containing images')
    parser.add_argument('--num_epochs', type=int, default=2000000,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configurations without running')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create experiment runner
    global runner
    runner = ExperimentRunner()

    # Base configuration
    base_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'num_epochs': args.num_epochs,
        'save_freq': args.save_freq,
        'num_workers': args.num_workers,
        'weight_decay': args.weight_decay
    }

    # Create experiment configurations
    configs = runner.create_experiment_configs(base_config)

    if args.dry_run:
        logging.info("Dry run mode - printing configurations:")
        for i, config in enumerate(configs):
            logging.info(f"\nExperiment {i + 1}:")
            for key, value in config.items():
                logging.info(f"  {key}: {value}")
        return

    # Log experiment configurations
    logging.info("Experiment configurations:")
    for i, config in enumerate(configs):
        logging.info(f"\nExperiment {i + 1}:")
        logging.info(f"  Device: {config['device']}")
        logging.info(f"  Output: {config['output_dir']}")
        logging.info(f"  Schedule: {config['schedule_type']}")
        logging.info(f"  Channels: {config['base_channels']}")
        logging.info(f"  Learning Rate: {config['learning_rate']}")
        logging.info(f"  Batch Size: {config['batch_size']}")
        logging.info(f"  Timesteps: {config['num_timesteps']}")
        logging.info(f"  Sample Frequency: {config['sample_freq']} epochs")

    # Check if we're in conda environment
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        logging.warning(
            "Not in conda environment. Make sure 'llm' environment is activated.")
    elif os.environ['CONDA_DEFAULT_ENV'] != 'llm':
        logging.warning(
            f"Current environment: {os.environ['CONDA_DEFAULT_ENV']}, expected 'llm'")

    # Run experiments
    logging.info("Starting parallel training experiments...")
    start_time = time.time()

    try:
        results = runner.run_all_experiments(configs)

        # Log results
        logging.info("\nExperiment Results:")
        successful = 0
        failed = 0

        for result in results:
            exp_id = result.get('exp_id', 'unknown')
            status = result.get('status', 'unknown')

            if status == 'success':
                successful += 1
                logging.info(f"  Experiment {exp_id}: SUCCESS")
            else:
                failed += 1
                logging.error(f"  Experiment {exp_id}: FAILED - {result}")

        total_time = time.time() - start_time
        logging.info(
            f"\nAll experiments completed in {total_time:.2f} seconds")
        logging.info(f"Successful: {successful}, Failed: {failed}")

        # Create summary
        summary = {
            'total_experiments': len(configs),
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'results': results,
            'configs': configs
        }

        # Save summary
        import json
        os.makedirs('./outputs', exist_ok=True)
        with open('./outputs/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(
            "Experiment summary saved to ./outputs/experiment_summary.json")

    except KeyboardInterrupt:
        logging.info("Experiments interrupted by user")
    except Exception as e:
        logging.error(f"Error running experiments: {str(e)}")
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
