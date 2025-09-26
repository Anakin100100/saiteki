import os
import shutil
import tempfile
import httpx
import time
from pathlib import Path
from loguru import logger
from .models import OptimizationTask
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


class TaskService:
    def __init__(self, api_base_url: str = "http://localhost:3000"):
        self.temp_dir = None
        self.api_base_url = api_base_url
        self.current_task_id = None
        self.log_buffer = []
        self.accumulated_logs = ""  # Add accumulated logs tracking
        self.log_file_path = None  # Track the evolution log file path
        self.synced_log_lines = 0  # Track how many lines have been synced
        
    def create_evaluate_py(self, task: OptimizationTask) -> str:
        """Create a generic evaluate.py file based on task data."""
        # Get the absolute path to the saiteki-worker directory
        worker_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        content = f'''"""
Generic evaluator for optimization task {task.id}
"""

import sys
import os
# Add the worker root directory to Python path so shinka can be imported
sys.path.insert(0, "{worker_root}")

import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval


{task.validateResultFunc}


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for experiment runs."""
    return {{}}


{task.generateMetricsFunc}


def main(program_path: str, results_dir: str):
    """Runs the optimization evaluation using shinka.eval."""
    print(f"Evaluating program: {{program_path}}")
    print(f"Saving results to: {{results_dir}}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator_with_context(r: List[Any]) -> Dict[str, Any]:
        return aggregate_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_optimization",
        num_runs=1,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_result,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {{error_msg}}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {{key}}: {{value}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic optimization evaluator")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
'''
        return content
    
    def create_initial_py(self, task: OptimizationTask) -> str:
        """Create a generic initial.py file based on task data."""
        content = f'''# EVOLVE-BLOCK-START
"""Generic optimization function for task {task.id}"""

import numpy as np


{task.optimizedFunc}


# EVOLVE-BLOCK-END
def run_optimization():
    centers, radii = optimize()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
'''
        return content
    
    def setup_task_environment(self, task: OptimizationTask) -> str:
        """Setup temporary directory with task files."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"task_{task.id}_")
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Write evaluate.py
        evaluate_content = self.create_evaluate_py(task)
        evaluate_path = os.path.join(self.temp_dir, "evaluate.py")
        with open(evaluate_path, 'w') as f:
            f.write(evaluate_content)
        
        # Write initial.py
        initial_content = self.create_initial_py(task)
        initial_path = os.path.join(self.temp_dir, "initial.py")
        with open(initial_path, 'w') as f:
            f.write(initial_content)
        
        print(f"Created task files in {self.temp_dir}")
        return self.temp_dir
    
    def cleanup_task_environment(self):
        """Clean up temporary directory and results."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
        
        # Also clean up results directory if it exists
        if os.path.exists("results"):
            shutil.rmtree("results")
            print("Cleaned up results directory")
    
    def update_task_logs(self, task_id: str, logs: str, running: bool = None):
        """Update optimization task logs via API."""
        try:
            # Accumulate logs instead of overwriting
            if logs:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                new_log_line = f"[{timestamp}] {logs}\n"
                self.accumulated_logs += new_log_line
            
            with httpx.Client() as client:
                data = {
                    "task_id": task_id,
                    "logs": self.accumulated_logs  # Send complete log history
                }
                if running is not None:
                    data["running"] = running
                    
                response = client.post(
                    f"{self.api_base_url}/internal/api/optimization/update_optimization_task_log",
                    json=data
                )
                response.raise_for_status()
                print(f"Updated logs for task {task_id}")
        except Exception as e:
            print(f"Failed to update task logs: {e}")
    
    def save_optimization_result(self, task_id: str, generation_num: int, 
                                     solution_code: str, combined_score: float, 
                                     public_metrics: dict):
        """Save optimization result via API."""
        try:
            with httpx.Client() as client:
                data = {
                    "optimization_task_id": task_id,
                    "generation_num": generation_num,
                    "solution_code": solution_code,
                    "combined_score": combined_score,
                    "public_metrics": public_metrics
                }
                print(f"Saving optimization result: {data}")
                
                response = client.post(
                    f"{self.api_base_url}/internal/api/optimization/create_optimization_result",
                    json=data
                )
                print(response.json())
                response.raise_for_status()
                print(f"Saved result for task {task_id}, generation {generation_num}")
        except Exception as e:
            print(f"Failed to save optimization result: {e}")
    
    def complete_optimization_task(self, task_id: str):
        """Mark optimization task as complete via API."""
        try:
            with httpx.Client() as client:
                data = {"task_id": task_id}
                
                response = client.post(
                    f"{self.api_base_url}/internal/api/optimization/complete_optimization_task",
                    json=data
                )
                response.raise_for_status()
                print(f"Marked task {task_id} as complete")
        except Exception as e:
            print(f"Failed to complete task: {e}")
    
    def flush_logs_to_buffer(self):
        """Flush current logs to buffer for API update."""
        # This would collect logs from loguru or other sources
        # For now, we'll use a simple approach
        if hasattr(logger, '_core') and logger._core.handlers:
            # Get recent log entries - this is a simplified approach
            # In practice, you might want to use a custom log handler
            log_content = f"Task {self.current_task_id} processing update\n"
            self.log_buffer.append(log_content)
    
    def sync_evolution_log(self, task_id: str, running: bool = None):
        """Synchronize evolution log file with API by reading new lines."""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return
            
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
            # Get only the new lines that haven't been synced
            new_lines = all_lines[self.synced_log_lines:]
            
            if new_lines:
                # Add new lines to accumulated logs
                new_log_content = ''.join(new_lines)
                self.accumulated_logs += new_log_content
                
                # Update synced line count
                self.synced_log_lines = len(all_lines)
                
                # Send updated logs to API
                with httpx.Client() as client:
                    data = {
                        "task_id": task_id,
                        "logs": self.accumulated_logs
                    }
                    if running is not None:
                        data["running"] = running
                        
                    response = client.post(
                        f"{self.api_base_url}/internal/api/optimization/update_optimization_task_log",
                        json=data
                    )
                    response.raise_for_status()
                    print(f"Synced {len(new_lines)} new log lines for task {task_id}")
                    
        except Exception as e:
            print(f"Failed to sync evolution log: {e}")
    
    def run_optimization_task(self, task: OptimizationTask):
        """Run the complete optimization task with setup and cleanup."""
        self.current_task_id = task.id
        self.accumulated_logs = ""  # Reset logs for new task
        self.synced_log_lines = 0  # Reset synced line count
        
        try:
            # Setup environment
            task_dir = self.setup_task_environment(task)
            
            # Set up evolution log file path
            self.log_file_path = os.path.join(task_dir, "results", "evolution_run.log")
            
            # Change to task directory
            original_cwd = os.getcwd()
            os.chdir(task_dir)
            
            # Create evolution configs (copied from run_evo.py)
            job_config = LocalJobConfig(eval_program_path=os.path.join(task_dir, "evaluate.py"))
            
            db_config = DatabaseConfig(
                db_path="evolution_db.sqlite",
                num_islands=2,
                archive_size=40,
                elite_selection_ratio=0.3,
                num_archive_inspirations=4,
                num_top_k_inspirations=2,
                migration_interval=10,
                migration_rate=0.1,
                island_elitism=True,
                parent_selection_strategy="weighted",
                parent_selection_lambda=10.0,
            )

            search_task_sys_msg = """You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Make sure that all circles are disjoint and lie inside the unit square.

Be creative and try to find a new solution better than the best known result."""
            
            evo_config = EvolutionConfig(
                task_sys_msg=search_task_sys_msg,
                patch_types=["diff", "full", "cross"],
                patch_type_probs=[0.6, 0.3, 0.1],
                num_generations=10,  # Reduced for faster iteration
                max_parallel_jobs=2,  # Reduced for worker
                max_patch_resamples=3,
                max_patch_attempts=3,
                job_type="local",
                language="python",
                llm_models=["gpt-5", "gpt-5-mini"],
                llm_kwargs=dict(
                    temperatures=[0.0, 0.5, 1.0],
                    reasoning_efforts=["auto", "low", "medium", "high"],
                    max_tokens=32768,
                ),
                meta_rec_interval=5,  # Reduced
                meta_llm_models=["gpt-5-mini"],
                meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
                embedding_model="text-embedding-3-small",
                code_embed_sim_threshold=0.995,
                novelty_llm_models=["gpt-5-mini"],
                novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
                llm_dynamic_selection=None,  # Disable dynamic selection to avoid numerical issues
                llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
                init_program_path=os.path.join(task_dir, "initial.py"),
                results_dir=os.path.join(task_dir, "results"),
            )
            
            # Create custom EvolutionRunner with task integration
            runner = TaskAwareEvolutionRunner(
                evo_config=evo_config,
                job_config=job_config,
                db_config=db_config,
                task_service=self,
                task_id=task.id
            )
            
            # Run evolution
            print(f"Starting evolution for task {task.id}")
            runner.run()
            
            print(f"Completed evolution for task {task.id}")
            
            # Final sync of evolution log
            self.sync_evolution_log(task_id=task.id, running=False)
            
        except Exception as e:
            print(f"Error running optimization task {task.id}: {e}")
            # Sync logs before marking as failed
            self.sync_evolution_log(task_id=task.id, running=False)
            raise
        finally:
            # Always ensure final log sync and task is marked as not running
            try:
                # Final sync of any remaining logs
                self.sync_evolution_log(task_id=task.id, running=False)
            except Exception as final_error:
                print(f"Failed to finalize task status: {final_error}")
            
            # Restore original directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            # Cleanup
            self.cleanup_task_environment()
            self.current_task_id = None
            self.accumulated_logs = ""  # Reset for next task
            self.log_file_path = None
            self.synced_log_lines = 0


class TaskAwareEvolutionRunner(EvolutionRunner):
    """Custom EvolutionRunner that integrates with TaskService for API calls."""
    
    def __init__(self, task_service: TaskService, task_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_service = task_service
        self.task_id = task_id
    
    def _process_completed_job(self, job):
        """Override to add result saving and log flushing."""
        print("BEFORE THE CALL")
        # Get values directly from the job object
        try:
            # Read the evaluated code from the job
            try:
                print("TRYING TO READ CODE")
                evaluated_code = Path(job.exec_fname).read_text(encoding="utf-8")
            except Exception as e:
                print(f"Could not read code for job {job.job_id}. Error: {e}")
                evaluated_code = ""
            
            # Get results from the job
            results = self.scheduler.get_job_results(job.job_id, job.results_dir)
            
            correct_val = False
            metrics_val = {}
            combined_score = 0.0
            public_metrics = {}
            
            if results:
                correct_val = results.get("correct", {}).get("correct", False)
                metrics_val = results.get("metrics", {})
                combined_score = metrics_val.get("combined_score", 0.0)
                public_metrics = metrics_val.get("public", {})
            
            # Save result via API using job data (synchronous)
            try:
                print("SAVING RESULT")
                self.task_service.save_optimization_result(
                    task_id=self.task_id,
                    generation_num=job.generation,
                    solution_code=evaluated_code,
                    combined_score=combined_score,
                    public_metrics=public_metrics
                )
                
                # Sync evolution log to get latest progress
                self.task_service.sync_evolution_log(
                    task_id=self.task_id,
                    running=True
                )
                
            except Exception as e:
                print(f"Failed to save result or update logs: {e}")
                
        except Exception as e:
            print(f"Failed to process completed job for API integration: {e}")

        print("AFTER THE CALL")

        super()._process_completed_job(job)
