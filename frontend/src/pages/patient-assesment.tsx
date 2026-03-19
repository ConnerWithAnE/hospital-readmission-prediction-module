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

    function handleResult(data: PredictionResult, values: Record<string, unknown>) {
        setResult(data)
        setPatientInput(values)
        sessionStorage.setItem(RESULT_KEY, JSON.stringify(data))
        sessionStorage.setItem(INPUT_KEY, JSON.stringify(values))
    }

    function handleClear() {
        setResult(null)
        setPatientInput({})
        sessionStorage.removeItem(RESULT_KEY)
        sessionStorage.removeItem(INPUT_KEY)
    }

    return (
        <div className="flex w-full h-full gap-4">
            <InfoWindow onResult={handleResult} onClear={handleClear} />
            <ScoreWindow result={result} patientInput={patientInput} />
        </div>
    )
}
