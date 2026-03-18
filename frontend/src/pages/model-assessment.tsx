import { useEffect, useState } from "react"

interface ModelStats {
    AUROC: string
    AUPRC: string
    best_f1_threshold: string
    classification_report: string
    confusion_matrix_png: string
    calibration_curve_png: string
}

export function ModelAssessmentPage() {
    const [metrics, setMetrics] = useState<ModelStats | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetch("/api/model_stats")
            .then(res => res.json())
            .then(data => { setMetrics(data); setLoading(false) })
            .catch(err => { setError(err.message); setLoading(false) })
    }, [])

    if (loading) return <div className="flex flex-1 h-full items-center justify-center text-muted-foreground text-sm">Loading model stats...</div>
    if (error) return <div className="flex flex-1 h-full items-center justify-center text-destructive text-sm">Error: {error}</div>
    if (!metrics) return null

    return (
        <div className="flex flex-col gap-6 p-6 w-full h-full overflow-y-auto">
            {/* Metric cards */}
            <div className="grid grid-cols-3 gap-4">
                <MetricCard label="AUROC" value={metrics.AUROC} description="Area under the ROC curve" />
                <MetricCard label="AUPRC" value={metrics.AUPRC} description="Area under precision-recall curve" />
                <MetricCard label="Best F1 Threshold" value={metrics.best_f1_threshold} description="Optimal decision threshold" />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-4">
                <div className="border-2 rounded-lg p-4 flex flex-col gap-2">
                    <h3 className="font-semibold text-sm">Confusion Matrix</h3>
                    <img
                        src={`data:image/png;base64,${metrics.confusion_matrix_png}`}
                        alt="Confusion Matrix"
                        className="w-full object-contain"
                    />
                </div>
                <div className="border-2 rounded-lg p-4 flex flex-col gap-2">
                    <h3 className="font-semibold text-sm">Calibration Curve</h3>
                    <img
                        src={`data:image/png;base64,${metrics.calibration_curve_png}`}
                        alt="Calibration Curve"
                        className="w-full object-contain"
                    />
                </div>
            </div>

            {/* Classification report */}
            <div className="border-2 rounded-lg p-4 flex flex-col gap-2">
                <h3 className="font-semibold text-sm">Classification Report</h3>
                <pre className="text-xs font-mono bg-muted rounded p-3 overflow-x-auto whitespace-pre">
                    {metrics.classification_report}
                </pre>
            </div>
        </div>
    )
}

function MetricCard({ label, value, description }: { label: string; value: string; description: string }) {
    return (
        <div className="border-2 rounded-lg p-4 flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">{description}</span>
            <span className="text-3xl font-bold">{value}</span>
            <span className="text-sm font-medium">{label}</span>
        </div>
    )
}
