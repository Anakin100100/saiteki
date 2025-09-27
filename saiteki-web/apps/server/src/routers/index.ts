import { protectedProcedure, publicProcedure } from "../lib/orpc";
import type { RouterClient } from "@orpc/server";
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
	optimization: optimizationRouter,
};
export type AppRouter = typeof appRouter;
export type AppRouterClient = RouterClient<typeof appRouter>;

export const internalRouter = baseInternalRouter;
export type InternalRouter = typeof internalRouter;
