import z from "zod";
import prisma from "../../db";
import { publicProcedure } from "../../lib/orpc";

const updateOptimizationTaskLogInput = z.object({
	task_id: z.string().min(1),
	logs: z.string(),
	running: z.boolean().optional(),
});

const createOptimizationResultInput = z.object({
	optimization_task_id: z.string().min(1),
	generation_num: z.number().int().min(0),
	solution_code: z.string().min(1),
	combined_score: z.number(),
	public_metrics: z.record(z.string(), z.number()),
});

const completeOptimizationTaskInput = z.object({
	task_id: z.string().min(1),
});

export const internalOptimizationRouter = {
	update_optimization_task_log: publicProcedure
		.input(updateOptimizationTaskLogInput)
		.handler(async ({ input }) => {
			return prisma.optimizationTask.update({
				where: {
					id: input.task_id,
				},
				data: {
					logs: input.logs,
					...(input.running === undefined
						? {}
						: {
							running: input.running,
						}),
				},
			});
		}),

	create_optimization_result: publicProcedure
		.input(createOptimizationResultInput)
		.handler(async ({ input }) => {
			return prisma.optimizationResult.create({
				data: {
					optimizationTaskId: input.optimization_task_id,
					generationNum: input.generation_num,
					solutionCode: input.solution_code,
					combinedScore: input.combined_score,
					publicMetrics: input.public_metrics,
				},
			});
		}),

	complete_optimization_task: publicProcedure
		.input(completeOptimizationTaskInput)
		.handler(async ({ input }) => {
			return prisma.optimizationTask.update({
				where: {
					id: input.task_id,
				},
				data: {
					running: false,
				},
			});
		}),
};
