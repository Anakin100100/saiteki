import { internalOptimizationRouter } from "./optimization";

export const internalRouter = {
	optimization: internalOptimizationRouter,
};

export type InternalRouter = typeof internalRouter;
