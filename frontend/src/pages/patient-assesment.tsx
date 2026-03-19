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

export function PatientAssessmentPage() {
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [patientInput, setPatientInput] = useState<Record<string, unknown>>({})

    function handleResult(data: PredictionResult, values: Record<string, unknown>) {
        setResult(data)
        setPatientInput(values)
    }

    return (
        <div className="flex w-full h-full gap-4">
            <InfoWindow onResult={handleResult} />
            <ScoreWindow result={result} patientInput={patientInput} />
        </div>
    )
}
