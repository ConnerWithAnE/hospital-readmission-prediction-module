import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip"
import type { PredictionResult } from "@/pages/patient-assesment"
import { Field, FieldLabel } from "@/components/ui/field"
import { Button } from "@/components/ui/button"
import { generatePdfReport } from "@/lib/generate-pdf"

interface SupplyGap {
    drug: { generic_name: string; model_field: string | null }
    quantity_on_hand: number
    reorder_level: number
    model_field: string
}

interface ScoreWindowProps {
    result: PredictionResult | null
    patientInput?: Record<string, unknown>
}

const RISK_COLORS: Record<string, string> = {
    very_low: "text-green-600",
    low: "text-green-500",
    moderate: "text-yellow-500",
    high: "text-red-500",
    very_high: "text-red-700",
}
const RISK_BADGE: Record<string, string> = {
    very_low: "text-green-700 bg-green-50 border-green-300",
    low: "text-green-600 bg-green-50 border-green-200",
    moderate: "text-yellow-600 bg-yellow-50 border-yellow-200",
    high: "text-orange-600 bg-orange-50 border-orange-200",
    very_high: "text-red-700 bg-red-50 border-red-300",
}
const RISK_HEX: Record<string, string> = { very_low: "#16a34a", low: "#22c55e", moderate: "#eab308", high: "#ef4444", very_high: "#b91c1c" }
const POSITIVE_COLOR = "#ef4444"
const NEGATIVE_COLOR = "#22c55e"

// Features that a clinician cannot change
const NON_MODIFIABLE = new Set(["age", "gender"])

interface PieSlice { name: string; value: number; positive: boolean }

// ── Pie Chart ────────────────────────────────────────────────────────────────
function SvgPieChart({ slices }: { slices: PieSlice[] }) {
    const [tooltip, setTooltip] = useState<{ x: number; y: number; slice: PieSlice; pct: number } | null>(null)
    const size = 220, cx = size / 2, cy = size / 2, r = 90
    const total = slices.reduce((s, d) => s + d.value, 0)
    let angle = -Math.PI / 2
    const paths = slices.map((slice, i) => {
        const start = angle
        const sweep = (slice.value / total) * 2 * Math.PI
        angle += sweep
        const x1 = cx + r * Math.cos(start), y1 = cy + r * Math.sin(start)
        const x2 = cx + r * Math.cos(angle), y2 = cy + r * Math.sin(angle)
        const large = sweep > Math.PI ? 1 : 0
        const pct = Math.round((slice.value / total) * 100)
        return { i, slice, pct, d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z` }
    })
    return (
        <div className="relative flex justify-center">
            <svg width={size} height={size}>
                {paths.map(({ i, slice, pct, d }) => (
                    <path key={i} d={d}
                        fill={slice.positive ? POSITIVE_COLOR : NEGATIVE_COLOR}
                        stroke="white" strokeWidth={1.5} opacity={0.85} style={{ cursor: "pointer" }}
                        onMouseEnter={e => setTooltip({ x: e.clientX, y: e.clientY, slice, pct })}
                        onMouseLeave={() => setTooltip(null)} />
                ))}
            </svg>
            {tooltip && (
                <div className="fixed z-50 rounded-md border bg-popover px-3 py-2 text-xs shadow-md pointer-events-none"
                    style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}>
                    <p className="font-medium capitalize">{tooltip.slice.name}</p>
                    <p className="text-muted-foreground">{tooltip.pct}% of total impact</p>
                    <p className="text-muted-foreground">Impact: {tooltip.slice.positive ? "+" : "-"}{tooltip.slice.value.toFixed(3)}</p>
                </div>
            )}
        </div>
    )
}

// ── Horizontal Bar Chart ─────────────────────────────────────────────────────
function BarChart({ slices }: { slices: PieSlice[] }) {
    const maxVal = Math.max(...slices.map(s => s.value))
    return (
        <div className="flex flex-col gap-2 w-full">
            {slices.map((s, i) => {
                const pct = (s.value / maxVal) * 100
                return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="w-32 text-wrap text-right capitalize text-muted-foreground truncate shrink-0">{s.name}</span>
                        <div className="flex flex-1 items-center h-5 bg-muted rounded overflow-hidden">
                            <div className="h-full rounded transition-all"
                                style={{ width: `${pct}%`, background: s.positive ? POSITIVE_COLOR : NEGATIVE_COLOR, opacity: 0.85 }} />
                        </div>
                        <span className="w-12 text-muted-foreground shrink-0">
                            {s.positive ? "+" : "-"}{s.value.toFixed(3)}
                        </span>
                    </div>
                )
            })}
        </div>
    )
}

// ── Top 3 Cards ──────────────────────────────────────────────────────────────
function TopThreeCards({ slices, total }: { slices: PieSlice[]; total: number }) {
    const top3 = slices.slice(0, 3)
    return (
        <div className="flex flex-col gap-3 w-full">
            {top3.map((s, i) => {
                const pct = Math.round((s.value / total) * 100)
                return (
                    <div key={i} className="flex items-center gap-3 rounded-lg border p-3">
                        <span className="text-2xl font-bold text-muted-foreground w-6">#{i + 1}</span>
                        <div className="flex flex-col flex-1 min-w-0">
                            <span className="text-sm font-medium capitalize truncate">{s.name}</span>
                            <span className="text-xs text-muted-foreground">{pct}% of total impact</span>
                        </div>
                        <span className="text-lg font-semibold shrink-0"
                            style={{ color: s.positive ? POSITIVE_COLOR : NEGATIVE_COLOR }}>
                            {s.positive ? "↑" : "↓"}
                        </span>
                    </div>
                )
            })}
        </div>
    )
}


// Medication field names used by the model
const MED_FIELDS = new Set([
    "insulin", "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone", "metformin_rosiglitazone",
    "metformin_pioglitazone",
])

// ── Score Window ─────────────────────────────────────────────────────────────
export default function ScoreWindow({ result, patientInput }: ScoreWindowProps) {
    const [supplyWarnings, setSupplyWarnings] = useState<SupplyGap[]>([])

    useEffect(() => {
        if (!result || !patientInput) { setSupplyWarnings([]); return }

        // Find which medications the patient is actively using
        const activeMeds = new Set<string>()
        for (const [key, val] of Object.entries(patientInput)) {
            if (MED_FIELDS.has(key) && val && val !== "No") {
                // Convert underscore to hyphen to match model_field in inventory
                activeMeds.add(key.replace(/_/g, "-"))
                activeMeds.add(key) // also keep underscore variant
            }
        }

        if (activeMeds.size === 0) { setSupplyWarnings([]); return }

        fetch("/api/inventory/supply-gaps")
            .then(r => r.json())
            .then((gaps: SupplyGap[]) => {
                const relevant = gaps.filter(g =>
                    activeMeds.has(g.model_field) || activeMeds.has(g.model_field.replace(/-/g, "_"))
                )
                setSupplyWarnings(relevant)
            })
            .catch(() => setSupplyWarnings([]))
    }, [result, patientInput])

    if (!result) {
        return (
            <div className="flex flex-1 items-center justify-center text-muted-foreground text-sm p-6 border-2 rounded-lg">
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
    const total = slices.reduce((s, d) => s + d.value, 0)

    function handleExportPdf() {
        generatePdfReport({
            risk_score: result!.risk_score,
            risk_category: result!.risk_category,
            contributing_factors: result!.contributing_factors,
            patientInput,
        })
    }

    return (
        <div className="flex flex-1 flex-col gap-4 p-6 border-2 rounded-lg">
            {supplyWarnings.length > 0 && (
                <div className="rounded-md border border-amber-400 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                    <span className="font-semibold">Low Inventory Warning:</span>{" "}
                    {supplyWarnings.map((g, i) => (
                        <span key={g.model_field}>
                            {i > 0 && ", "}
                            {g.drug.generic_name}
                            <span className="text-amber-600"> ({g.quantity_on_hand} left)</span>
                        </span>
                    ))}
                </div>
            )}
            <div className="flex items-center justify-between">
                <div className="flex-1" />
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <span
                                className={`text-4xl font-bold capitalize cursor-default px-4 py-1 rounded-lg border-2 ${RISK_BADGE[result.risk_category] ?? ""}`}
                            >
                                {result.risk_category.replace("_", " ")} Risk
                            </span>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="text-sm">
                            <p className="font-semibold">{percentage}% probability</p>
                            <p className="text-muted-foreground text-xs">30-day readmission</p>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
                <div className="flex-1 flex justify-end">
                    <Button variant="outline" size="sm" onClick={handleExportPdf}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        Export PDF
                    </Button>
                </div>
            </div>
            <div className="w-full">
                <div className="flex-1 gap-4 text-xs text-muted-foreground mb-2">
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full bg-red-500" />Increases risk
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full bg-green-500" />Decreases risk
                    </span>
                </div>
                <div className="flex p-4">
                    <SvgPieChart slices={slices} />
                    <BarChart slices={slices} />
                </div>
                <FieldLabel className="text-xl font-medium pb-3">Top Contributing Factors</FieldLabel>
                <TopThreeCards slices={slices} total={total} />  
            </div>

            
        </div>
    )
}
