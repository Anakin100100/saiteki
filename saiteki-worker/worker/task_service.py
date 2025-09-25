import os
import shutil
import tempfile
import httpx
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
        
    def create_evaluate_py(self, task: OptimizationTask) -> str:
        """Create a generic evaluate.py file based on task data."""
        content = f'''"""
Generic evaluator for optimization task {task.id}
"""

import os
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
    """Run the optimization function."""
    result = optimize()
    return result
'''
        return content
    
    def setup_task_environment(self, task: OptimizationTask) -> str:
        """Setup temporary directory with task files."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"task_{task.id}_")
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
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
        
        logger.info(f"Created task files in {self.temp_dir}")
        return self.temp_dir
    
    def cleanup_task_environment(self):
        """Clean up temporary directory and results."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
        
        # Also clean up results directory if it exists
        if os.path.exists("results"):
            shutil.rmtree("results")
            logger.info("Cleaned up results directory")
    
    async def update_task_logs(self, task_id: str, logs: str, running: bool = None):
        """Update optimization task logs via API."""
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "task_id": task_id,
                    "logs": logs
                }
                if running is not None:
                    data["running"] = running
                    
                response = await client.post(
                    f"{self.api_base_url}/internal/rpc/optimization/update_optimization_task_log",
                    json=data
                )
                response.raise_for_status()
                logger.debug(f"Updated logs for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to update task logs: {e}")
    
    async def save_optimization_result(self, task_id: str, generation_num: int, 
                                     solution_code: str, combined_score: float, 
                                     public_metrics: dict):
        """Save optimization result via API."""
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "optimization_task_id": task_id,
                    "generation_num": generation_num,
                    "solution_code": solution_code,
                    "combined_score": combined_score,
                    "public_metrics": public_metrics
                }
                
                response = await client.post(
                    f"{self.api_base_url}/internal/rpc/optimization/create_optimization_result",
                    json=data
                )
                response.raise_for_status()
                logger.info(f"Saved result for task {task_id}, generation {generation_num}")
        except Exception as e:
            logger.error(f"Failed to save optimization result: {e}")
    
    async def complete_optimization_task(self, task_id: str):
        """Mark optimization task as complete via API."""
        try:
            async with httpx.AsyncClient() as client:
                data = {"task_id": task_id}
                
                response = await client.post(
                    f"{self.api_base_url}/internal/rpc/optimization/complete_optimization_task",
                    json=data
                )
                response.raise_for_status()
                logger.info(f"Marked task {task_id} as complete")
        except Exception as e:
            logger.error(f"Failed to complete task: {e}")
    
    def flush_logs_to_buffer(self):
        """Flush current logs to buffer for API update."""
        # This would collect logs from loguru or other sources
        # For now, we'll use a simple approach
        if hasattr(logger, '_core') and logger._core.handlers:
            # Get recent log entries - this is a simplified approach
            # In practice, you might want to use a custom log handler
            log_content = f"Task {self.current_task_id} processing update\n"
            self.log_buffer.append(log_content)
    
    def run_optimization_task(self, task: OptimizationTask):
        """Run the complete optimization task with setup and cleanup."""
        self.current_task_id = task.id
        
        try:
            # Setup environment
            task_dir = self.setup_task_environment(task)
            
            # Change to task directory
            original_cwd = os.getcwd()
            os.chdir(task_dir)
            
            # Create evolution configs (copied from run_evo.py)
            job_config = LocalJobConfig(eval_program_path="evaluate.py")
            
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
            
            evo_config = EvolutionConfig(
                task_sys_msg="You are an expert optimizer. Improve the optimization function to maximize the result.",
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
                llm_dynamic_selection="ucb1",
                llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
                init_program_path="initial.py",
                results_dir="results",
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
            logger.info(f"Starting evolution for task {task.id}")
            runner.run()
            
            logger.info(f"Completed evolution for task {task.id}")
            
            # Mark task as complete
            import asyncio
            asyncio.run(self.complete_optimization_task(task.id))
            
        except Exception as e:
            logger.error(f"Error running optimization task {task.id}: {e}")
            # Try to mark task as failed
            try:
                import asyncio
                asyncio.run(self.complete_optimization_task(task.id))
            except:
                pass
            raise
        finally:
            # Restore original directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            # Cleanup
            self.cleanup_task_environment()
            self.current_task_id = None


class TaskAwareEvolutionRunner(EvolutionRunner):
    """Custom EvolutionRunner that integrates with TaskService for API calls."""
    
    def __init__(self, task_service: TaskService, task_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_service = task_service
        self.task_id = task_id
    
    def _process_completed_job(self, job):
        """Override to add result saving and log flushing."""
        # Call parent implementation first
        super()._process_completed_job(job)
        
        # Get the program that was just added
        try:
            # Find the most recent program for this generation
            programs = self.db.get_programs_by_generation(job.generation)
            if programs:
                # Get the program that was just added (should be the last one)
                latest_program = max(programs, key=lambda p: p.id)
                
                # Save result via API
                import asyncio
                asyncio.run(self.task_service.save_optimization_result(
                    task_id=self.task_id,
                    generation_num=job.generation,
                    solution_code=latest_program.code,
                    combined_score=latest_program.combined_score,
                    public_metrics=latest_program.public_metrics or {}
                ))
                
                # Flush logs
                log_content = f"Generation {job.generation} completed. Score: {latest_program.combined_score}"
                asyncio.run(self.task_service.update_task_logs(
                    task_id=self.task_id,
                    logs=log_content,
                    running=True
                ))
                
        except Exception as e:
            logger.error(f"Failed to save result or update logs: {e}")
