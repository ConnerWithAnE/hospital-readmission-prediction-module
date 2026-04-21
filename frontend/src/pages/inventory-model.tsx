import { useEffect, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ResultRow {
    model: string
    MAE: number
    RMSE: number
    MAPE: number
    granularity: string
}

interface ModelStats {
    best_model: string
    best_type: string
    granularity: string
    MAE: number
    RMSE: number
    MAPE: number
    feature_count: number
    results_summary: ResultRow[]
}

interface FeatureImportance {
    feature: string
    importance: number
}

interface PredictionRow {
    product_name: string
    state: string
    quarter?: number
    month?: number
    target: number
    predicted: number
    error: number
}

interface TopAggregate {
    product_name?: string
    state?: string
    total_predicted: number
}

interface Summary {
    unique_drugs: number
    unique_states: number
    total_test_rows: number
    predicted_total: number
    actual_total: number
    top_drugs: TopAggregate[]
    top_states: TopAggregate[]
}

function fmt(n: number | string | undefined): string {
    if (n === undefined || n === "" || n === null) return "—"
    const num = typeof n === "string" ? parseFloat(n) : n
    if (isNaN(num)) return String(n)
    if (Math.abs(num) >= 1000) return num.toLocaleString(undefined, { maximumFractionDigits: 0 })
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export function InventoryModelPage() {
    const [stats, setStats] = useState<ModelStats | null>(null)
    const [summary, setSummary] = useState<Summary | null>(null)
    const [features, setFeatures] = useState<FeatureImportance[]>([])
    const [chart, setChart] = useState<string | null>(null)
    const [predictions, setPredictions] = useState<PredictionRow[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    // Prediction controls
    const [mode, setMode] = useState<"top" | "bottom" | "random" | "largest_error">("top")
    const [limit, setLimit] = useState(50)
    const [filterState, setFilterState] = useState("")
    const [filterDrug, setFilterDrug] = useState("")
    const [granularity, setGranularity] = useState<"monthly" | "quarterly">("monthly")

    // Initial fetch for stats + summary + chart
    useEffect(() => {
        setLoading(true)
        Promise.all([
            fetch("/api/inventory/model/stats").then(r => r.ok ? r.json() : Promise.reject(r.statusText)),
            fetch("/api/inventory/model/summary").then(r => r.ok ? r.json() : null),
            fetch("/api/inventory/model/comparison-chart").then(r => r.ok ? r.json() : null),
        ])
            .then(([s, sm, ch]) => {
                setStats(s)
                setSummary(sm)
                if (ch) setChart(ch.png_base64)
            })
            .catch(e => setError(String(e)))
            .finally(() => setLoading(false))
    }, [])

    // Feature importance — refetch on granularity change
    useEffect(() => {
        fetch(`/api/inventory/model/feature-importance?granularity=${granularity}`)
            .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
            .then(d => setFeatures(d.features))
            .catch(() => setFeatures([]))
    }, [granularity])

    // Predictions — refetch on controls change
    useEffect(() => {
        const params = new URLSearchParams({ mode, limit: String(limit) })
        if (filterState) params.set("state", filterState)
        if (filterDrug) params.set("drug", filterDrug)
        fetch(`/api/inventory/model/predictions?${params}`)
            .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
            .then(d => setPredictions(d.rows))
            .catch(() => setPredictions([]))
    }, [mode, limit, filterState, filterDrug])

    if (loading) {
        return <div className="flex flex-1 h-full items-center justify-center text-muted-foreground text-sm">
            Loading inventory model...
        </div>
    }
    if (error) {
        return <div className="flex flex-1 h-full items-center justify-center text-destructive text-sm">
            Error: {error} — did you run <code className="mx-1 rounded bg-muted px-1">train.py</code>?
        </div>
    }
    if (!stats) return null

    const maxImportance = features.length > 0 ? Math.max(...features.map(f => f.importance)) : 1

    return (
        <div className="flex flex-col gap-6 p-6 w-full h-full overflow-y-auto">
            <div>
                <h1 className="text-2xl font-bold">Inventory Demand Model</h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Cross-sectional demand forecasting trained on Medicare Part D + FDA Drug Utilization (2023)
                </p>
            </div>

            {/* Headline metrics */}
            <div className="grid grid-cols-4 gap-4">
                <MetricCard label="Best Model" value={stats.best_model} description={`@ ${stats.granularity}`} />
                <MetricCard label="MAE" value={fmt(stats.MAE)} description="Mean Absolute Error (units)" />
                <MetricCard label="RMSE" value={fmt(stats.RMSE)} description="Root Mean Squared Error" />
                <MetricCard label="MAPE" value={`${fmt(stats.MAPE)}%`} description="Mean Absolute Pct Error" />
            </div>

            <Tabs defaultValue="overview">
                <TabsList className="w-fit">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="features">Feature Importance</TabsTrigger>
                    <TabsTrigger value="predictions">Predictions</TabsTrigger>
                    <TabsTrigger value="aggregates">Top Drugs & States</TabsTrigger>
                </TabsList>

                {/* ── Overview ─────────────────────────────────────────── */}
                <TabsContent value="overview">
                    <div className="flex flex-col gap-4">
                        {/* Results summary table */}
                        <div className="border-2 rounded-lg p-4">
                            <h3 className="font-semibold text-sm mb-3">Results by Granularity</h3>
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b bg-muted/50">
                                        <th className="text-left p-2 font-semibold">Model</th>
                                        <th className="text-left p-2 font-semibold">Granularity</th>
                                        <th className="text-right p-2 font-semibold">MAE</th>
                                        <th className="text-right p-2 font-semibold">RMSE</th>
                                        <th className="text-right p-2 font-semibold">MAPE</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.results_summary.map((r, i) => {
                                        const isBest = r.granularity === stats.granularity
                                        return (
                                            <tr key={i} className={`border-b last:border-b-0 ${isBest ? "bg-green-50 font-semibold" : ""}`}>
                                                <td className="p-2">{r.model}</td>
                                                <td className="p-2 capitalize">{r.granularity}</td>
                                                <td className="p-2 text-right">{fmt(r.MAE)}</td>
                                                <td className="p-2 text-right">{fmt(r.RMSE)}</td>
                                                <td className="p-2 text-right">{fmt(r.MAPE)}%</td>
                                            </tr>
                                        )
                                    })}
                                </tbody>
                            </table>
                        </div>

                        {/* Comparison chart */}
                        {chart && (
                            <div className="border-2 rounded-lg p-4 flex flex-col gap-2">
                                <h3 className="font-semibold text-sm">Model Comparison</h3>
                                <img
                                    src={`data:image/png;base64,${chart}`}
                                    alt="Model Comparison"
                                    className="w-full object-contain"
                                />
                            </div>
                        )}

                        {/* Summary pills */}
                        {summary && (
                            <div className="grid grid-cols-3 gap-4">
                                <MetricCard label="Unique Drugs" value={fmt(summary.unique_drugs)} description="in test set" />
                                <MetricCard label="Unique States" value={fmt(summary.unique_states)} description="in test set" />
                                <MetricCard label="Test Rows" value={fmt(summary.total_test_rows)} description="drug-state-period records" />
                            </div>
                        )}
                    </div>
                </TabsContent>

                {/* ── Feature Importance ───────────────────────────────── */}
                <TabsContent value="features">
                    <div className="flex flex-col gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Granularity:</span>
                            <Button
                                variant={granularity === "monthly" ? "default" : "outline"}
                                size="sm"
                                onClick={() => setGranularity("monthly")}
                            >Monthly</Button>
                            <Button
                                variant={granularity === "quarterly" ? "default" : "outline"}
                                size="sm"
                                onClick={() => setGranularity("quarterly")}
                            >Quarterly</Button>
                        </div>

                        <div className="border-2 rounded-lg p-4">
                            <h3 className="font-semibold text-sm mb-3">
                                Feature Importance (GBM · {granularity})
                            </h3>
                            <div className="flex flex-col gap-1.5">
                                {features.filter(f => f.importance > 0).map(f => (
                                    <div key={f.feature} className="flex items-center gap-3">
                                        <div className="w-52 text-xs font-mono truncate" title={f.feature}>{f.feature}</div>
                                        <div className="flex-1 bg-muted rounded-sm h-4 overflow-hidden">
                                            <div
                                                className="bg-blue-500 h-full"
                                                style={{ width: `${(f.importance / maxImportance) * 100}%` }}
                                            />
                                        </div>
                                        <div className="w-20 text-right text-xs font-mono text-muted-foreground">
                                            {f.importance.toFixed(4)}
                                        </div>
                                    </div>
                                ))}
                                {features.filter(f => f.importance === 0).length > 0 && (
                                    <p className="text-xs text-muted-foreground mt-2">
                                        {features.filter(f => f.importance === 0).length} additional features had zero importance.
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                </TabsContent>

                {/* ── Predictions ──────────────────────────────────────── */}
                <TabsContent value="predictions">
                    <div className="flex flex-col gap-4">
                        {/* Controls */}
                        <div className="flex items-center gap-3 flex-wrap border-2 rounded-lg p-3">
                            <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground mr-1">Sort:</span>
                                {(["top", "bottom", "largest_error", "random"] as const).map(m => (
                                    <Button
                                        key={m}
                                        variant={mode === m ? "default" : "outline"}
                                        size="xs"
                                        onClick={() => setMode(m)}
                                    >
                                        {m === "top" ? "Highest volume" : m === "bottom" ? "Lowest volume" : m === "largest_error" ? "Largest error" : "Random"}
                                    </Button>
                                ))}
                            </div>
                            <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground mr-1">Limit:</span>
                                {[25, 50, 100, 250].map(n => (
                                    <Button
                                        key={n}
                                        variant={limit === n ? "default" : "outline"}
                                        size="xs"
                                        onClick={() => setLimit(n)}
                                    >{n}</Button>
                                ))}
                            </div>
                            <Input
                                className="max-w-[140px]"
                                placeholder="State (e.g. CA)"
                                value={filterState}
                                onChange={e => setFilterState(e.target.value)}
                            />
                            <Input
                                className="max-w-[220px]"
                                placeholder="Drug contains..."
                                value={filterDrug}
                                onChange={e => setFilterDrug(e.target.value)}
                            />
                        </div>

                        <div className="border-2 rounded-lg overflow-hidden">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-muted/50 border-b">
                                        <th className="text-left p-2 font-semibold">Drug</th>
                                        <th className="text-left p-2 font-semibold">State</th>
                                        <th className="text-center p-2 font-semibold">Period</th>
                                        <th className="text-right p-2 font-semibold">Actual</th>
                                        <th className="text-right p-2 font-semibold">Predicted</th>
                                        <th className="text-right p-2 font-semibold">Error</th>
                                        <th className="text-right p-2 font-semibold">% Error</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictions.length === 0 ? (
                                        <tr>
                                            <td colSpan={7} className="p-6 text-center text-sm text-muted-foreground">
                                                No predictions match the current filter.
                                            </td>
                                        </tr>
                                    ) : predictions.map((p, i) => {
                                        const actual = Number(p.target)
                                        const predicted = Number(p.predicted)
                                        const err = Number(p.error)
                                        const pctErr = actual !== 0 ? (err / actual) * 100 : 0
                                        const period = p.month ?? p.quarter ?? "—"
                                        return (
                                            <tr key={i} className="border-b last:border-b-0 hover:bg-muted/30">
                                                <td className="p-2 font-medium truncate max-w-[300px]" title={p.product_name}>{p.product_name}</td>
                                                <td className="p-2 font-mono text-xs">{p.state}</td>
                                                <td className="p-2 text-center text-xs font-mono">{period}</td>
                                                <td className="p-2 text-right font-mono text-xs">{fmt(actual)}</td>
                                                <td className="p-2 text-right font-mono text-xs">{fmt(predicted)}</td>
                                                <td className="p-2 text-right font-mono text-xs">{fmt(err)}</td>
                                                <td className={`p-2 text-right font-mono text-xs ${pctErr > 50 ? "text-red-600" : pctErr > 20 ? "text-amber-600" : "text-green-600"}`}>
                                                    {pctErr.toFixed(1)}%
                                                </td>
                                            </tr>
                                        )
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </TabsContent>

                {/* ── Top drugs & states ───────────────────────────────── */}
                <TabsContent value="aggregates">
                    {summary ? (
                        <div className="grid grid-cols-2 gap-4">
                            <div className="border-2 rounded-lg p-4">
                                <h3 className="font-semibold text-sm mb-3">Top 10 Drugs by Predicted Volume</h3>
                                <AggregateList
                                    rows={summary.top_drugs}
                                    labelKey="product_name"
                                />
                            </div>
                            <div className="border-2 rounded-lg p-4">
                                <h3 className="font-semibold text-sm mb-3">Top 10 States by Predicted Volume</h3>
                                <AggregateList
                                    rows={summary.top_states}
                                    labelKey="state"
                                />
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground">Summary unavailable.</p>
                    )}
                </TabsContent>
            </Tabs>
        </div>
    )
}

function MetricCard({ label, value, description }: { label: string; value: string; description: string }) {
    return (
        <div className="border-2 rounded-lg p-4 flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">{description}</span>
            <span className="text-2xl font-bold truncate">{value}</span>
            <span className="text-sm font-medium">{label}</span>
        </div>
    )
}

function AggregateList({ rows, labelKey }: { rows: TopAggregate[]; labelKey: "product_name" | "state" }) {
    if (rows.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>
    const max = Math.max(...rows.map(r => r.total_predicted))
    return (
        <div className="flex flex-col gap-1.5">
            {rows.map((r, i) => (
                <div key={i} className="flex items-center gap-3">
                    <div className="w-48 text-xs font-mono truncate" title={String(r[labelKey])}>{r[labelKey]}</div>
                    <div className="flex-1 bg-muted rounded-sm h-4 overflow-hidden">
                        <div
                            className="bg-emerald-500 h-full"
                            style={{ width: `${(r.total_predicted / max) * 100}%` }}
                        />
                    </div>
                    <div className="w-28 text-right text-xs font-mono text-muted-foreground">
                        {fmt(r.total_predicted)}
                    </div>
                </div>
            ))}
        </div>
    )
}