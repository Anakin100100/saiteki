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
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { orpc } from "@/utils/orpc";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Code2 } from "lucide-react";
import Plot from "react-plotly.js";
import { useEffect, useMemo, useState } from "react";

function parseMetrics(value: unknown) {
	if (!value || typeof value !== "object") {
		return {} as Record<string, number>;
	}

	return Object.fromEntries(
		Object.entries(value as Record<string, unknown>).flatMap(([key, metricValue]) => {
			return typeof metricValue === "number" ? [[key, metricValue]] : [];
		}),
	) as Record<string, number>;
}

export const Route = createFileRoute("/optimization/$taskId")({
	component: OptimizationTaskDetailRoute,
});

function OptimizationTaskDetailRoute() {
	const { taskId } = Route.useParams();

	const tasksQuery = useQuery(orpc.optimization.query_optimization_tasks.queryOptions());
	const resultsQuery = useQuery(
		orpc.optimization.query_optimization_results.queryOptions({
			input: {
				task_id: taskId,
			},
		}),
	);

	const task = useMemo(() => {
		return tasksQuery.data?.find((entry) => entry.id === taskId);
	}, [tasksQuery.data, taskId]);

	const results = useMemo(() => {
		return (
			resultsQuery.data?.map((result) => ({
				...result,
				metrics: parseMetrics(result.publicMetrics),
			})) ?? []
		).sort((a, b) => a.generationNum - b.generationNum);
	}, [resultsQuery.data]);

	const metricOptions = useMemo(() => {
		const options = new Set<string>();
		options.add("combined_score");
		for (const result of results) {
			Object.keys(result.metrics).forEach((metricKey) => {
				options.add(metricKey);
			});
		}
		return Array.from(options);
	}, [results]);

	const [selectedMetric, setSelectedMetric] = useState<string>("combined_score");

	useEffect(() => {
		if (metricOptions.length === 0) {
			return;
		}

		if (!metricOptions.includes(selectedMetric)) {
			setSelectedMetric(metricOptions[0]);
		}
	}, [metricOptions, selectedMetric]);

	const chartData = useMemo(() => {
		if (results.length === 0) {
			return [];
		}

		const yValues = results.map((result) => {
			if (selectedMetric === "combined_score") {
				return result.combinedScore;
			}
			return result.metrics[selectedMetric] ?? null;
		});

		const label = selectedMetric === "combined_score" ? "Combined Score" : selectedMetric;

		return [
			{
				x: results.map((result) => result.generationNum),
				y: yValues,
				type: "scatter" as const,
				mode: "lines+markers" as const,
				line: {
					color: "hsl(var(--primary))",
				},
				marker: {
					color: "hsl(var(--primary))",
				},
				hovertemplate: "Generation %{x}<br>Value %{y:.4f}<extra></extra>",
				name: label,
			},
		];
	}, [results, selectedMetric]);

	const chartLayout = useMemo(() => {
		const metricLabel = selectedMetric === "combined_score" ? "Combined Score" : selectedMetric;
		return {
			autosize: true,
			margin: { t: 48, r: 24, b: 56, l: 60 },
			paper_bgcolor: "transparent",
			plot_bgcolor: "transparent",
			font: { color: "currentColor" },
			title: {
				text: metricLabel,
			},
			xaxis: {
				title: {
					text: "Generation",
				},
				tickmode: "linear" as const,
				zeroline: false,
			},
			yaxis: {
				title: {
					text: metricLabel,
				},
				zeroline: false,
			},
		};
	}, [selectedMetric]);

	const noTask = !tasksQuery.isLoading && !task;
	const noResults = !resultsQuery.isLoading && results.length === 0;

	return (
		<div className="mx-auto flex w-full max-w-6xl flex-col gap-6 py-10">
			<div className="flex flex-col gap-2">
				<Button asChild variant="ghost" className="w-fit">
					<Link to="/optimization">
						<ArrowLeft className="mr-2 h-4 w-4" /> Back to tasks
					</Link>
				</Button>
				<div className="flex flex-wrap items-center gap-3">
					<h1 className="text-3xl font-semibold">Task {taskId.slice(0, 8)}</h1>
					{task ? (
						<Badge variant={task.running ? "default" : "secondary"}>
							{task.running ? "Running" : "Completed"}
						</Badge>
					) : null}
				</div>
				{task ? (
					<p className="text-muted-foreground text-sm">
						Created {new Date(task.createdAt).toLocaleString()} · Last updated
						 {" "}
						{new Date(task.updatedAt).toLocaleString()}
					</p>
				) : null}
			</div>

			{noTask ? (
				<Card>
					<CardHeader>
						<CardTitle>Task not found</CardTitle>
					</CardHeader>
					<CardContent>
						<p className="text-sm text-muted-foreground">
							We couldn't find this optimization task. It may have been deleted or
							you might not have access to it.
						</p>
					</CardContent>
				</Card>
			) : null}

			<Card>
				<CardHeader className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<CardTitle>Performance metrics</CardTitle>
						<CardDescription>
							Track how this task evolves across generations.
						</CardDescription>
					</div>
					{metricOptions.length > 0 ? (
						<Select value={selectedMetric} onValueChange={setSelectedMetric}>
							<SelectTrigger className="w-52">
								<SelectValue placeholder="Pick a metric" />
							</SelectTrigger>
							<SelectContent>
								{metricOptions.map((metric) => (
									<SelectItem key={metric} value={metric}>
										{metric === "combined_score"
											? "Combined Score"
											: metric}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					) : null}
				</CardHeader>
				<CardContent className="h-[420px]">
					{resultsQuery.isLoading ? (
						<div className="flex h-full items-center justify-center">
							<LoaderSkeleton />
						</div>
					) : results.length === 0 ? (
						<div className="flex h-full items-center justify-center text-sm text-muted-foreground">
							No results yet. The worker will populate this view when generations are available.
						</div>
					) : (
						<Plot
							data={chartData}
							layout={chartLayout}
							config={{ displayModeBar: false, responsive: true }}
							className="h-full w-full"
							style={{ width: "100%", height: "100%" }}
						/>
					)}
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Generation history</CardTitle>
					<CardDescription>
						Review combined scores, metrics, and generated solution code.
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					{resultsQuery.isLoading ? (
						<LoaderSkeleton rows={4} />
					) : noResults ? (
						<p className="text-sm text-muted-foreground">
							No optimization results available for this task yet.
						</p>
					) : (
						<>
							<Table>
								<TableHeader>
									<TableRow>
										<TableHead>Generation</TableHead>
										<TableHead>Combined score</TableHead>
										<TableHead>Metrics</TableHead>
									</TableRow>
								</TableHeader>
								<TableBody>
									{results.map((result) => (
										<TableRow key={result.id}>
											<TableCell>{result.generationNum}</TableCell>
											<TableCell>{result.combinedScore.toFixed(4)}</TableCell>
											<TableCell>
												<div className="flex flex-wrap gap-2">
													{Object.entries(result.metrics).map(([metricKey, metricValue]) => (
														<Badge key={metricKey} variant="outline">
															{metricKey}: {metricValue.toFixed(4)}
														</Badge>
														))}
												</div>
											</TableCell>
										</TableRow>
									))}
								</TableBody>
							</Table>

							<div className="space-y-4">
								{results.map((result) => (
									<Card key={`${result.id}-code`}>
										<CardHeader className="flex flex-col gap-1">
											<CardTitle className="flex items-center gap-2 text-base">
												<Code2 className="h-4 w-4" /> Generation {result.generationNum}
											</CardTitle>
										</CardHeader>
										<CardContent>
											<ScrollArea className="max-h-64 rounded-md border bg-muted/40 p-4 text-xs">
												<pre className="whitespace-pre-wrap break-words">
													{result.solutionCode}
												</pre>
											</ScrollArea>
										</CardContent>
									</Card>
								))}
							</div>
						</>
					)}
				</CardContent>
			</Card>

			{task?.logs ? (
				<Card>
					<CardHeader>
						<CardTitle>Worker logs</CardTitle>
						<CardDescription>
							Live updates and diagnostics shared by the optimization worker.
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ScrollArea className="max-h-64 rounded-md border bg-muted/40 p-4 text-xs">
							<pre className="whitespace-pre-wrap break-words">
								{task.logs}
							</pre>
						</ScrollArea>
					</CardContent>
				</Card>
			) : null}
		</div>
	);
}

function LoaderSkeleton({ rows = 6 }: { rows?: number }) {
	return (
		<div className="flex w-full flex-col gap-2">
			{Array.from({ length: rows }).map((_, index) => (
				<div
					key={index}
					className="animate-pulse rounded-md bg-muted/60"
					style={{ height: 16 }}
				/>
			))}
		</div>
	);
}
