import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { orpc } from "@/utils/orpc";

export const Route = createFileRoute("/optimization")({
	component: OptimizationListRoute,
});

function OptimizationListRoute() {
	const tasksQuery = useQuery(orpc.optimization.query_optimization_tasks.queryOptions());

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
					<CardTitle>Your tasks</CardTitle>
					<CardDescription>
						Click any task to open the detailed results dashboard.
					</CardDescription>
				</CardHeader>
				<CardContent>
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
