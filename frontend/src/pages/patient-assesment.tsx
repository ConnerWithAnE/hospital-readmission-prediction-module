import { useState } from "react"
import InfoWindow from "@/components/patient-assessment/info-window";
import ScoreWindow from "@/components/patient-assessment/score-window";

export interface PredictionResult {
    risk_score: number
    risk_category: "very_low" | "low" | "moderate" | "high" | "very_high"
    contributing_factors: {
        feature: string
        value: number
        impact: number
    }[]
}

export interface MedImpactScenario {
    status: string
    risk_score: number
    delta: number
}

export interface MedImpact {
    name: string
    label: string
    current_status: string
    scenarios: MedImpactScenario[]
    max_reduction: number
    max_increase: number
    best_status: string
}

export interface MedImpactResult {
    baseline_risk: number
    baseline_category: string
    medications: MedImpact[]
}

const RESULT_KEY = "patient-assessment-result"
const INPUT_KEY = "patient-assessment-input"

function loadSession<T>(key: string, fallback: T): T {
    try {
        const saved = sessionStorage.getItem(key)
        return saved ? JSON.parse(saved) : fallback
    } catch { return fallback }
}

export function PatientAssessmentPage() {
    const [result, setResult] = useState<PredictionResult | null>(() => loadSession(RESULT_KEY, null))
    const [patientInput, setPatientInput] = useState<Record<string, unknown>>(() => loadSession(INPUT_KEY, {}))
    const [medImpact, setMedImpact] = useState<MedImpactResult | null>(null)
    const [medImpactLoading, setMedImpactLoading] = useState(false)

    function handleResult(data: PredictionResult, values: Record<string, unknown>) {
        setResult(data)
        setPatientInput(values)
        setMedImpact(null) // stale after re-assessment
        sessionStorage.setItem(RESULT_KEY, JSON.stringify(data))
        sessionStorage.setItem(INPUT_KEY, JSON.stringify(values))
    }

    function handleClear() {
        setResult(null)
        setPatientInput({})
        setMedImpact(null)
        sessionStorage.removeItem(RESULT_KEY)
        sessionStorage.removeItem(INPUT_KEY)
    }

    function handleRequestMedImpact() {
        if (!patientInput || Object.keys(patientInput).length === 0) return
        setMedImpactLoading(true)
        fetch("/api/predict/med-impact", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patientInput),
        })
            .then(r => r.json())
            .then(data => setMedImpact(data))
            .catch(console.error)
            .finally(() => setMedImpactLoading(false))
    }

    return (
        <div className="flex w-full h-full gap-4">
            <InfoWindow onResult={handleResult} onClear={handleClear} />
            <ScoreWindow
                result={result}
                patientInput={patientInput}
                medImpact={medImpact}
                medImpactLoading={medImpactLoading}
                onRequestMedImpact={handleRequestMedImpact}
            />
        </div>
    )
}
