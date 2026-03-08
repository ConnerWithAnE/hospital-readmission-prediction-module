import { useState } from "react"
import type { PredictionResult } from "@/pages/patient-assesment"

interface ScoreWindowProps {
    result: PredictionResult | null
}

const RISK_COLORS = {
    low: "text-green-500",
    moderate: "text-yellow-500",
    high: "text-red-500",
}

const POSITIVE_COLOR = "#ef4444"
const NEGATIVE_COLOR = "#22c55e"

interface PieSlice { name: string; value: number; positive: boolean }

function SvgPieChart({ slices }: { slices: PieSlice[] }) {
    const [tooltip, setTooltip] = useState<{ x: number; y: number; slice: PieSlice; pct: number } | null>(null)
    const size = 220
    const cx = size / 2
    const cy = size / 2
    const r = 90
    const total = slices.reduce((s, d) => s + d.value, 0)

    let angle = -Math.PI / 2
    const paths = slices.map((slice, i) => {
        const start = angle
        const sweep = (slice.value / total) * 2 * Math.PI
        angle += sweep
        const x1 = cx + r * Math.cos(start)
        const y1 = cy + r * Math.sin(start)
        const x2 = cx + r * Math.cos(angle)
        const y2 = cy + r * Math.sin(angle)
        const large = sweep > Math.PI ? 1 : 0
        const pct = Math.round((slice.value / total) * 100)
        return { i, slice, pct, d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z` }
    })

    return (
        <div className="relative">
            <svg width={size} height={size}>
                {paths.map(({ i, slice, pct, d }) => (
                    <path key={i} d={d}
                        fill={slice.positive ? POSITIVE_COLOR : NEGATIVE_COLOR}
                        stroke="white" strokeWidth={1.5} opacity={0.85}
                        style={{ cursor: "pointer" }}
                        onMouseEnter={e => setTooltip({ x: e.clientX, y: e.clientY, slice, pct })}
                        onMouseLeave={() => setTooltip(null)}
                    />
                ))}
            </svg>
            {tooltip && (
                <div className="fixed z-50 rounded-md border bg-popover px-3 py-2 text-xs shadow-md pointer-events-none"
                    style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}>
                    <p className="font-medium capitalize">{tooltip.slice.name}</p>
                    <p className="text-muted-foreground">{tooltip.pct}% of total impact</p>
                    <p className="text-muted-foreground">
                        Impact: {tooltip.slice.positive ? "+" : "-"}{tooltip.slice.value.toFixed(3)}
                    </p>
                </div>
            )}
        </div>
    )
}

export default function ScoreWindow({ result }: ScoreWindowProps) {
    if (!result) {
        return (
            <div className="flex flex-1 items-center justify-center text-muted-foreground text-sm">
                Run an assessment to see results.
            </div>
        )
    }

    const percentage = Math.round(result.risk_score * 100)
    const slices: PieSlice[] = result.contributing_factors.slice(0, 8).map(f => ({
        name: f.feature.replace(/_/g, " "),
        value: Math.abs(f.impact),
        positive: f.impact > 0,
    }))

    return (
        <div className="flex flex-1 flex-col gap-6 p-6 border-2 rounded-lg">
            <div className="flex flex-col items-center gap-1">
                <span className="text-6xl font-bold">{percentage}%</span>
                <span className={`text-lg font-medium capitalize ${RISK_COLORS[result.risk_category]}`}>
                    {result.risk_category} Risk
                </span>
                <span className="text-sm text-muted-foreground">30-day readmission probability</span>
            </div>

            <div className="flex flex-col gap-2">
                <span className="text-sm font-medium">Contributing Factors</span>
                <div className="flex gap-4 text-xs text-muted-foreground mb-1">
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                        Increases risk
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
                        Decreases risk
                    </span>
                </div>
                <div className="flex justify-center">
                    <SvgPieChart slices={slices} />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                    {slices.map((s, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                            <span className="inline-block w-2 h-2 rounded-full shrink-0"
                                style={{ background: s.positive ? POSITIVE_COLOR : NEGATIVE_COLOR }} />
                            <span className="capitalize text-muted-foreground">{s.name}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
