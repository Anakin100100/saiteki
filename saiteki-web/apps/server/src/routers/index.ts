import { protectedProcedure, publicProcedure } from "../lib/orpc";
import type { RouterClient } from "@orpc/server";
import { todoRouter } from "./todo";
import { optimizationRouter } from "./optimization";
import { internalRouter as baseInternalRouter } from "./internal";

export const appRouter = {
	healthCheck: publicProcedure.handler(() => {
		return "OK";
	}),
	privateData: protectedProcedure.handler(({ context }) => {
		return {
			message: "This is private",
			user: context.session?.user,
		};
	}),
	todo: todoRouter,
	optimization: optimizationRouter,
};
export type AppRouter = typeof appRouter;
export type AppRouterClient = RouterClient<typeof appRouter>;

export const internalRouter = baseInternalRouter;
export type InternalRouter = typeof internalRouter;
