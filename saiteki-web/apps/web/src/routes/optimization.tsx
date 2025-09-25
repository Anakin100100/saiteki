import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Form,
	FormControl,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { createFileRoute, Link, Outlet, useRouterState } from "@tanstack/react-router";
import { Loader2, PlusCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { orpc } from "@/utils/orpc";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

export const Route = createFileRoute("/optimization")({
	component: OptimizationListRoute,
});

const createTaskSchema = z.object({
	optimized_func: z.string().min(1, "Please provide the optimization function."),
	validate_result_func: z
		.string()
		.min(1, "Please provide the validation logic."),
	generate_metrics_func: z
		.string()
		.min(1, "Please provide the metrics generation logic."),
});

function OptimizationListRoute() {
	const location = useRouterState({
		select: (state) => state.location,
	});

	const normalizedPath =
		location.pathname === "/"
			? "/"
			: location.pathname.replace(/\/+$/, "");

	const viewingDetail =
		normalizedPath.startsWith("/optimization/") && normalizedPath !== "/optimization";

	if (viewingDetail) {
		return <Outlet />;
	}

	const tasksQuery = useQuery(orpc.optimization.query_optimization_tasks.queryOptions());
	const [isFormOpen, setIsFormOpen] = useState(false);

	const form = useForm<z.infer<typeof createTaskSchema>>({
		resolver: zodResolver(createTaskSchema),
		defaultValues: {
			optimized_func: "",
			validate_result_func: "",
			generate_metrics_func: "",
		},
	});

	const createTaskMutation = useMutation(
		orpc.optimization.create_optimization_task.mutationOptions({
			onSuccess: () => {
				tasksQuery.refetch();
				form.reset();
				setIsFormOpen(false);
			},
		}),
	);

	const hasRunningTask = tasksQuery.data?.some((task) => task.running) ?? false;

	const isFormDisabled = hasRunningTask || tasksQuery.isLoading;

	const handleSubmit = form.handleSubmit((values) => {
		if (isFormDisabled) {
			return;
		}

		createTaskMutation.mutate(values);
	});

	const sortedTasks = useMemo(() => {
		if (!tasksQuery.data) {
			return [];
		}

		return [...tasksQuery.data].sort((a, b) => {
			return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
		});
	}, [tasksQuery.data]);

	return (
		<div className="mx-auto flex w-full max-w-5xl flex-col gap-6 py-10">
			<div className="flex flex-col gap-2">
				<h1 className="text-3xl font-semibold">Optimization Tasks</h1>
				<p className="text-muted-foreground">
					Review your optimization tasks and open a task to inspect its results and
					run history.
				</p>
			</div>

			<Card>
				<CardHeader>
					<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
						<div className="space-y-1">
							<CardTitle>Your tasks</CardTitle>
							<CardDescription>
								Click any task to open the detailed results dashboard.
							</CardDescription>
						</div>
						<Button
							onClick={() => setIsFormOpen((prev) => !prev)}
							variant="secondary"
							className="bg-emerald-600 text-emerald-50 hover:bg-emerald-500"
							disabled={isFormDisabled}
						>
							<PlusCircle className="mr-2 h-4 w-4" />
							Create
						</Button>
					</div>
				</CardHeader>
				<CardContent>
					{isFormOpen && (
						<div className="mb-6 rounded-lg border border-emerald-700 bg-emerald-950/40 p-4">
							<Form {...form}>
								<form onSubmit={handleSubmit} className="space-y-4">
									<FormField
										control={form.control}
										name="optimized_func"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Optimized function</FormLabel>
												<FormControl>
													<Textarea
														placeholder="function optimize(...) { ... }"
														rows={4}
														disabled={isFormDisabled || createTaskMutation.isPending}
														{...field}
													/>
												</FormControl>
												<FormMessage />
											</FormItem>
										)}
									/>
									<FormField
										control={form.control}
										name="validate_result_func"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Validate result function</FormLabel>
												<FormControl>
													<Textarea
														placeholder="function validate(...) { ... }"
														rows={4}
														disabled={isFormDisabled || createTaskMutation.isPending}
														{...field}
													/>
												</FormControl>
												<FormMessage />
											</FormItem>
										)}
									/>
									<FormField
										control={form.control}
										name="generate_metrics_func"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Generate metrics function</FormLabel>
												<FormControl>
													<Textarea
														placeholder="function metrics(...) { ... }"
														rows={4}
														disabled={isFormDisabled || createTaskMutation.isPending}
														{...field}
													/>
												</FormControl>
												<FormMessage />
											</FormItem>
										)}
									/>
									<div className="flex items-center justify-end gap-2">
										<Button
											type="button"
											variant="ghost"
											onClick={() => setIsFormOpen(false)}
											disabled={createTaskMutation.isPending}
										>
											Cancel
										</Button>
										<Button
											type="submit"
											className="bg-emerald-600 text-emerald-50 hover:bg-emerald-500"
											disabled={createTaskMutation.isPending || isFormDisabled}
										>
											{createTaskMutation.isPending ? (
												<Loader2 className="h-4 w-4 animate-spin" />
											) : (
												<span>Create task</span>
											)}
										</Button>
									</div>
									{hasRunningTask && (
										<p className="text-sm text-emerald-300">
											You already have a running task. Please wait for it to finish
											before creating a new one.
										</p>
									)}
								</form>
							</Form>
						</div>
					)}
				
					{tasksQuery.isLoading ? (
						<div className="flex h-24 items-center justify-center">
							<Loader2 className="h-6 w-6 animate-spin" />
						</div>
					) : sortedTasks.length === 0 ? (
						<div className="text-center text-sm text-muted-foreground">
							You don't have any optimization tasks yet.
						</div>
					) : (
						<div className="grid gap-4 sm:grid-cols-2">
							{sortedTasks.map((task) => {
								const created = new Date(task.createdAt);
								const updated = new Date(task.updatedAt);
								const statusLabel = task.running ? "Running" : "Completed";

								return (
									<Card key={task.id} className="border shadow-sm">
										<CardContent className="flex flex-col gap-4 p-4">
											<div className="flex items-center justify-between">
												<div className="flex flex-col gap-1">
													<span className="text-lg font-medium">
														Task {task.id.slice(0, 8)}
													</span>
													<div className="flex gap-4 text-xs text-muted-foreground">
														<span>
															Created {created.toLocaleString()}
														</span>
														<span>
															Updated {updated.toLocaleString()}
														</span>
													</div>
												</div>
												<Badge variant={task.running ? "default" : "secondary"}>
													{statusLabel}
												</Badge>
											</div>
											<Button asChild variant="outline">
												<Link
													to="/optimization/$taskId"
													params={{ taskId: task.id }}
												>
													Open details
												</Link>
											</Button>
										</CardContent>
									</Card>
								);
							})}
						</div>
					)}
				</CardContent>
			</Card>
		</div>
	);
}
