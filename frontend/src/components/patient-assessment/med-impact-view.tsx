import { Button } from "@/components/ui/button"
import type { MedImpactResult } from "@/pages/patient-assesment"

interface MedImpactViewProps {
    data: MedImpactResult | null
    loading: boolean
    onRequest: () => void
    hasAssessment: boolean
}

const STATUSES = ["No", "Steady", "Down", "Up"]

export default function MedImpactView({ data, loading, onRequest, hasAssessment }: MedImpactViewProps) {
    if (!hasAssessment) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm gap-2">
                <p>Run a patient assessment first to enable medication impact analysis.</p>
            </div>
        )
    }

    if (!data && !loading) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-3">
                <p className="text-sm text-muted-foreground">
                    Analyze how changing each medication's dose status would affect this patient's readmission risk.
                </p>
                <Button onClick={onRequest}>Run Medication Impact Analysis</Button>
            </div>
        )
    }

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
                <div className="animate-spin h-6 w-6 border-2 border-current border-t-transparent rounded-full" />
                <p className="text-sm">Analyzing 69 medication scenarios...</p>
            </div>
        )
    }

    if (!data) return null

    const baselinePct = Math.round(data.baseline_risk * 100)

    // Find the most impactful medication
    const mostImpactful = data.medications[0] // already sorted by max_reduction
    const bestRiskPct = mostImpactful
        ? Math.round((data.baseline_risk + mostImpactful.max_reduction) * 100)
        : baselinePct

    // Top 8 for tornado chart
    const tornadoData = data.medications
        .filter(m => Math.abs(m.max_reduction) > 0.0001 || Math.abs(m.max_increase) > 0.0001)
        .slice(0, 8)

    const maxAbsDelta = Math.max(
        ...tornadoData.map(m => Math.max(Math.abs(m.max_reduction), Math.abs(m.max_increase))),
        0.001
    )

    return (
        <div className="flex flex-col gap-4 overflow-y-auto">
            {/* Re-run button */}
            <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                    Baseline risk: <span className="font-semibold text-foreground">{baselinePct}%</span>
                </span>
                <Button variant="outline" size="sm" onClick={onRequest} disabled={loading}>
                    Re-run Analysis
                </Button>
            </div>

            {/* Summary banner */}
            {mostImpactful && Math.abs(mostImpactful.max_reduction) > 0.0001 && (
                <div className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-800">
                    <span className="font-semibold">Most impactful change:</span>{" "}
                    Switching <span className="font-semibold">{mostImpactful.label}</span> from{" "}
                    <span className="font-mono">{mostImpactful.current_status}</span> to{" "}
                    <span className="font-mono">{mostImpactful.best_status}</span> would change risk from{" "}
                    <span className="font-semibold">{baselinePct}%</span> to{" "}
                    <span className="font-semibold">{bestRiskPct}%</span>{" "}
                    <span className="text-green-700">
                        ({mostImpactful.max_reduction > 0 ? "+" : ""}{Math.round(mostImpactful.max_reduction * 100)} pp)
                    </span>
                </div>
            )}

            {/* Tornado chart */}
            {tornadoData.length > 0 && (
                <div>
                    <h3 className="text-sm font-semibold mb-2">Impact Range by Medication</h3>
                    <TornadoChart data={tornadoData} maxDelta={maxAbsDelta} />
                </div>
            )}

            {/* Detail table */}
            <div>
                <h3 className="text-sm font-semibold mb-2">All Medications — Risk by Dose Status</h3>
                <div className="border rounded-lg overflow-hidden">
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="bg-muted/50 border-b">
                                <th className="text-left p-2 font-semibold">Medication</th>
                                <th className="text-center p-2 font-semibold">Current</th>
                                {STATUSES.map(s => (
                                    <th key={s} className="text-center p-2 font-semibold">{s}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {data.medications.map(med => (
                                <tr key={med.name} className="border-b last:border-b-0 hover:bg-muted/30">
                                    <td className="p-2 font-medium">{med.label}</td>
                                    <td className="p-2 text-center text-muted-foreground font-mono text-xs">
                                        {med.current_status}
                                    </td>
                                    {STATUSES.map(status => {
                                        const isCurrent = status === med.current_status
                                        const scenario = med.scenarios.find(s => s.status === status)

                                        if (isCurrent) {
                                            return (
                                                <td key={status} className="p-2 text-center bg-muted/40 text-muted-foreground font-mono">
                                                    {Math.round(data.baseline_risk * 100)}%
                                                </td>
                                            )
                                        }

                                        if (!scenario) return <td key={status} className="p-2 text-center">—</td>

                                        const riskPct = Math.round(scenario.risk_score * 100)
                                        const deltaPct = Math.round(scenario.delta * 100)
                                        const isGood = scenario.delta < -0.005
                                        const isBad = scenario.delta > 0.005

                                        return (
                                            <td
                                                key={status}
                                                className={`p-2 text-center font-mono ${
                                                    isGood ? "text-green-700 bg-green-50" :
                                                    isBad ? "text-red-700 bg-red-50" :
                                                    "text-muted-foreground"
                                                }`}
                                            >
                                                {riskPct}%
                                                {deltaPct !== 0 && (
                                                    <span className="text-[10px] ml-0.5">
                                                        ({deltaPct > 0 ? "+" : ""}{deltaPct})
                                                    </span>
                                                )}
                                            </td>
                                        )
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

// ── Tornado Chart ────────────────────────────────────────────────────────────
function TornadoChart({ data, maxDelta }: { data: { label: string; max_reduction: number; max_increase: number }[]; maxDelta: number }) {
    const barH = 22
    const gap = 4
    const labelW = 140
    const chartW = 300
    const centerX = labelW + chartW / 2
    const totalW = labelW + chartW + 60
    const totalH = data.length * (barH + gap) + 4

    return (
        <svg width="100%" viewBox={`0 0 ${totalW} ${totalH}`} className="max-w-xl">
            {/* Center line */}
            <line x1={centerX} y1={0} x2={centerX} y2={totalH} stroke="#cbd5e1" strokeWidth={1} />

            {data.map((med, i) => {
                const y = i * (barH + gap) + 2
                const reductionW = (Math.abs(med.max_reduction) / maxDelta) * (chartW / 2)
                const increaseW = (Math.abs(med.max_increase) / maxDelta) * (chartW / 2)

                return (
                    <g key={med.label}>
                        {/* Label */}
                        <text x={labelW - 6} y={y + barH / 2 + 4} textAnchor="end" className="text-xs fill-current" fontSize={11}>
                            {med.label}
                        </text>

                        {/* Reduction bar (green, extends left) */}
                        {reductionW > 0.5 && (
                            <rect
                                x={centerX - reductionW}
                                y={y}
                                width={reductionW}
                                height={barH}
                                rx={3}
                                fill="#22c55e"
                                opacity={0.8}
                            />
                        )}

                        {/* Increase bar (red, extends right) */}
                        {increaseW > 0.5 && (
                            <rect
                                x={centerX}
                                y={y}
                                width={increaseW}
                                height={barH}
                                rx={3}
                                fill="#ef4444"
                                opacity={0.8}
                            />
                        )}

                        {/* Delta labels */}
                        {Math.abs(med.max_reduction) > 0.0001 && (
                            <text x={centerX - reductionW - 4} y={y + barH / 2 + 4} textAnchor="end" fontSize={10} fill="#16a34a">
                                {Math.round(med.max_reduction * 100)}pp
                            </text>
                        )}
                        {Math.abs(med.max_increase) > 0.0001 && (
                            <text x={centerX + increaseW + 4} y={y + barH / 2 + 4} textAnchor="start" fontSize={10} fill="#dc2626">
                                +{Math.round(med.max_increase * 100)}pp
                            </text>
                        )}
                    </g>
                )
            })}
        </svg>
    )
}
