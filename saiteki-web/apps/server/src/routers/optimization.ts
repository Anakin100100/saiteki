import { ORPCError } from "@orpc/server";
import z from "zod";
import prisma from "../db";
import { protectedProcedure } from "../lib/orpc";
import { publishJson } from "../lib/nats";

const createOptimizationTaskInput = z.object({
	optimized_func: z.string().min(1),
	validate_result_func: z.string().min(1),
	generate_metrics_func: z.string().min(1),
});

const queryOptimizationResultsInput = z.object({
	task_id: z.string().min(1),
});

export const optimizationRouter = {
	create_optimization_task: protectedProcedure
		.input(createOptimizationTaskInput)
		.handler(async ({ input, context }) => {
			const userId = context.session?.user.id;

			if (!userId) {
				throw new ORPCError("UNAUTHORIZED");
			}

			const existingTask = await prisma.optimizationTask.findFirst({
				where: {
					userId,
					running: true,
				},
			});

			if (existingTask) {
				throw new ORPCError("BAD_REQUEST", {
					message: "You already have a running optimization task.",
				});
			}

			const task = await prisma.optimizationTask.create({
				data: {
					running: true,
					userId,
					optimizedFunc: input.optimized_func,
					validateResultFunc: input.validate_result_func,
					generateMetricsFunc: input.generate_metrics_func,
				},
			});

			try {
				await publishJson("created_optimization_tasks", {
					id: task.id,
					running: task.running,
					userId: task.userId,
					optimizedFunc: task.optimizedFunc,
					validateResultFunc: task.validateResultFunc,
					generateMetricsFunc: task.generateMetricsFunc,
					createdAt: task.createdAt.toISOString(),
					updatedAt: task.updatedAt.toISOString(),
					logs: task.logs,
				});
			} catch (error) {
				console.error("Failed to publish created optimization task", error);
			}

			return task;
		}),

	query_optimization_tasks: protectedProcedure.handler(async ({ context }) => {
		const userId = context.session?.user.id;

		if (!userId) {
			throw new ORPCError("UNAUTHORIZED");
		}

		return await prisma.optimizationTask.findMany({
			where: {
				userId,
			},
			orderBy: {
				createdAt: "desc",
			},
		});
	}),

	query_optimization_results: protectedProcedure
		.input(queryOptimizationResultsInput)
		.handler(async ({ input, context }) => {
			const userId = context.session?.user.id;

			if (!userId) {
				throw new ORPCError("UNAUTHORIZED");
			}

			const task = await prisma.optimizationTask.findFirst({
				where: {
					id: input.task_id,
					userId,
				},
				select: {
					id: true,
				},
			});

			if (!task) {
				throw new ORPCError("NOT_FOUND", {
					message: "Optimization task not found.",
				});
			}

			return await prisma.optimizationResult.findMany({
				where: {
					optimizationTaskId: input.task_id,
				},
				orderBy: {
					generationNum: "asc",
				},
			});
		}),
};
