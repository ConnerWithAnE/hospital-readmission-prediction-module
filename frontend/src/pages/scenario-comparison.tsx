import { useEffect, useState } from "react"

interface PredictionResult {
    risk_score: number
    risk_category: "very_low" | "low" | "moderate" | "high" | "very_high"
    contributing_factors: { feature: string; value: number; impact: number }[]
}

interface Scenario {
    name: string
    description: string
    input: Record<string, unknown>
    result?: PredictionResult | null
    error?: string
    loading?: boolean
}

const SCENARIOS: Scenario[] = [
    {
        name: "Young Elective — Low Risk",
        description: "25 y/o Caucasian woman, elective surgery, no comorbidities, short stay",
        input: {
            age: 25,
            gender: "female",
            race: "Caucasian",
            time_in_hospital: 2,
            admission_type: "elective",
            admission_source: "physician_referral",
            discharge_group: "Home",
            num_lab_procedures: 10,
            num_procedures: 1,
            num_medications: 3,
            number_diagnoses: 2,
            number_inpatient: 0,
            number_outpatient: 1,
            number_emergency: 0,
            comorbidities: [],
            a1c_result: "not_measured",
            max_glu_serum: "not_measured",
            diabetes_med: false,
            med_change: false,
            insulin: "No",
            metformin: "No",
            diag_category: "musculoskeletal",
            n_active_meds: 0,
            n_med_changes: 0,
        },
    },
    {
        name: "Elderly Emergency — High Risk",
        description: "70 y/o Caucasian woman, emergency admission, CHF + renal disease, long stay",
        input: {
            age: 70,
            gender: "female",
            race: "Caucasian",
            time_in_hospital: 9,
            admission_type: "emergency",
            admission_source: "emergency",
            discharge_group: "Home",
            num_lab_procedures: 45,
            num_procedures: 3,
            num_medications: 18,
            number_diagnoses: 9,
            number_inpatient: 3,
            number_outpatient: 5,
            number_emergency: 2,
            comorbidities: [
                "congestive_heart_failure",
                "renal_disease",
                "diabetes_complicated",
                "fluid_electrolyte_disorders",
            ],
            a1c_result: ">8",
            max_glu_serum: ">300",
            diabetes_med: true,
            med_change: true,
            insulin: "Up",
            metformin: "Steady",
            diag_category: "circulatory",
            n_active_meds: 3,
            n_med_changes: 2,
        },
    },
    {
        name: "Middle-Aged Diabetic — Moderate Risk",
        description: "55 y/o African American man, urgent admission, diabetes on insulin, moderate stay",
        input: {
            age: 55,
            gender: "male",
            race: "AfricanAmerican",
            time_in_hospital: 5,
            admission_type: "urgent",
            admission_source: "physician_referral",
            discharge_group: "Home",
            num_lab_procedures: 30,
            num_procedures: 2,
            num_medications: 12,
            number_diagnoses: 6,
            number_inpatient: 1,
            number_outpatient: 3,
            number_emergency: 1,
            comorbidities: ["diabetes_complicated", "hypertension_uncomplicated"],
            a1c_result: ">7",
            max_glu_serum: ">200",
            diabetes_med: true,
            med_change: false,
            insulin: "Steady",
            metformin: "Steady",
            diag_category: "endocrine",
            n_active_meds: 2,
            n_med_changes: 0,
        },
    },
    {
        name: "Frequent Readmitter",
        description: "65 y/o Hispanic man, emergency, 5 prior inpatient visits, multiple comorbidities",
        input: {
            age: 65,
            gender: "male",
            race: "Hispanic",
            time_in_hospital: 7,
            admission_type: "emergency",
            admission_source: "emergency",
            discharge_group: "care_facility",
            num_lab_procedures: 50,
            num_procedures: 4,
            num_medications: 20,
            number_diagnoses: 9,
            number_inpatient: 5,
            number_outpatient: 8,
            number_emergency: 4,
            comorbidities: [
                "congestive_heart_failure",
                "chronic_pulmonary_disease",
                "renal_disease",
                "depression",
                "diabetes_complicated",
                "fluid_electrolyte_disorders",
            ],
            a1c_result: ">8",
            max_glu_serum: ">300",
            diabetes_med: true,
            med_change: true,
            insulin: "Up",
            metformin: "Down",
            diag_category: "respiratory",
            n_active_meds: 4,
            n_med_changes: 3,
        },
    },
    {
        name: "Healthy Elective — Minimal Risk",
        description: "40 y/o Asian man, elective, no prior visits, no comorbidities",
        input: {
            age: 40,
            gender: "male",
            race: "Asian",
            time_in_hospital: 1,
            admission_type: "elective",
            admission_source: "physician_referral",
            discharge_group: "Home",
            num_lab_procedures: 5,
            num_procedures: 1,
            num_medications: 2,
            number_diagnoses: 1,
            number_inpatient: 0,
            number_outpatient: 0,
            number_emergency: 0,
            comorbidities: [],
            a1c_result: "Norm",
            max_glu_serum: "Norm",
            diabetes_med: false,
            med_change: false,
            insulin: "No",
            metformin: "No",
            diag_category: "endocrine",
            n_active_meds: 0,
            n_med_changes: 0,
        },
    },
    {
        name: "Post-Transfer Elderly",
        description: "80 y/o Caucasian woman, transferred from another facility, coagulopathy + weight loss",
        input: {
            age: 80,
            gender: "female",
            race: "Caucasian",
            time_in_hospital: 11,
            admission_type: "emergency",
            admission_source: "transfer",
            discharge_group: "care_facility",
            num_lab_procedures: 60,
            num_procedures: 2,
            num_medications: 15,
            number_diagnoses: 8,
            number_inpatient: 2,
            number_outpatient: 2,
            number_emergency: 3,
            comorbidities: [
                "congestive_heart_failure",
                "coagulopathy",
                "weight_loss",
                "renal_disease",
                "deficiency_anemias",
            ],
            a1c_result: ">7",
            max_glu_serum: ">200",
            diabetes_med: true,
            med_change: true,
            insulin: "Up",
            metformin: "No",
            diag_category: "circulatory",
            n_active_meds: 2,
            n_med_changes: 1,
        },
    },
]

const RISK_COLORS: Record<string, string> = {
    very_low: "text-green-700 bg-green-50 border-green-300",
    low: "text-green-600 bg-green-50 border-green-200",
    moderate: "text-yellow-600 bg-yellow-50 border-yellow-200",
    high: "text-orange-600 bg-orange-50 border-orange-200",
    very_high: "text-red-700 bg-red-50 border-red-300",
}

export function ScenarioComparisonPage() {
    const [scenarios, setScenarios] = useState<Scenario[]>(
        SCENARIOS.map((s) => ({ ...s, loading: true }))
    )

    useEffect(() => {
        SCENARIOS.forEach((scenario, i) => {
            fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(scenario.input),
            })
                .then((res) => res.json())
                .then((result) => {
                    setScenarios((prev) =>
                        prev.map((s, j) =>
                            j === i ? { ...s, result, loading: false } : s
                        )
                    )
                })
                .catch((err) => {
                    setScenarios((prev) =>
                        prev.map((s, j) =>
                            j === i
                                ? { ...s, error: err.message, loading: false }
                                : s
                        )
                    )
                })
        })
    }, [])

    return (
        <div className="flex flex-col gap-6 p-6 w-full h-full overflow-y-auto">
            <div>
                <h1 className="text-2xl font-bold">Scenario Comparison</h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Predefined patient scenarios showing how the model scores
                    different risk profiles
                </p>
            </div>

            {/* Summary table */}
            <div className="border-2 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="bg-muted/50 border-b">
                            <th className="text-left p-3 font-semibold">
                                Scenario
                            </th>
                            <th className="text-left p-3 font-semibold">
                                Description
                            </th>
                            <th className="text-center p-3 font-semibold">
                                Risk Score
                            </th>
                            <th className="text-center p-3 font-semibold">
                                Category
                            </th>
                            <th className="text-left p-3 font-semibold">
                                Top Factors
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {scenarios.map((scenario, i) => (
                            <tr
                                key={i}
                                className="border-b last:border-b-0 hover:bg-muted/30"
                            >
                                <td className="p-3 font-medium">
                                    {scenario.name}
                                </td>
                                <td className="p-3 text-muted-foreground">
                                    {scenario.description}
                                </td>
                                <td className="p-3 text-center">
                                    {scenario.loading ? (
                                        <span className="text-muted-foreground">
                                            ...
                                        </span>
                                    ) : scenario.error ? (
                                        <span className="text-destructive text-xs">
                                            Error
                                        </span>
                                    ) : (
                                        <span className="font-bold text-lg">
                                            {(
                                                (scenario.result
                                                    ?.risk_score ?? 0) * 100
                                            ).toFixed(1)}
                                            %
                                        </span>
                                    )}
                                </td>
                                <td className="p-3 text-center">
                                    {scenario.result && (
                                        <span
                                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${RISK_COLORS[scenario.result.risk_category] ?? ""}`}
                                        >
                                            {scenario.result.risk_category.replace("_", " ")}
                                        </span>
                                    )}
                                </td>
                                <td className="p-3">
                                    {scenario.result && (
                                        <div className="flex flex-col gap-0.5">
                                            {scenario.result.contributing_factors
                                                .slice(0, 3)
                                                .map((f, fi) => (
                                                    <span
                                                        key={fi}
                                                        className="text-xs"
                                                    >
                                                        <span className="font-medium">
                                                            {f.feature}
                                                        </span>{" "}
                                                        <span
                                                            className={
                                                                f.impact > 0
                                                                    ? "text-red-500"
                                                                    : "text-green-500"
                                                            }
                                                        >
                                                            (
                                                            {f.impact > 0
                                                                ? "+"
                                                                : ""}
                                                            {f.impact.toFixed(
                                                                3
                                                            )}
                                                            )
                                                        </span>
                                                    </span>
                                                ))}
                                        </div>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Detail cards */}
            <div className="grid grid-cols-2 gap-4">
                {scenarios.map((scenario, i) => (
                    <ScenarioCard key={i} scenario={scenario} />
                ))}
            </div>
        </div>
    )
}

function ScenarioCard({ scenario }: { scenario: Scenario }) {
    const { input, result } = scenario

    return (
        <div className="border-2 rounded-lg p-4 flex flex-col gap-3">
            <div className="flex items-start justify-between">
                <div>
                    <h3 className="font-semibold">{scenario.name}</h3>
                    <p className="text-xs text-muted-foreground">
                        {scenario.description}
                    </p>
                </div>
                {result && (
                    <span
                        className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${RISK_COLORS[result.risk_category] ?? ""}`}
                    >
                        {(result.risk_score * 100).toFixed(1)}%{" "}
                        {result.risk_category.replace("_", " ")}
                    </span>
                )}
                {scenario.loading && (
                    <span className="text-xs text-muted-foreground">
                        Loading...
                    </span>
                )}
                {scenario.error && (
                    <span className="text-xs text-destructive">
                        {scenario.error}
                    </span>
                )}
            </div>

            {/* Key input summary */}
            <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
                <KeyVal label="Age" value={input.age} />
                <KeyVal label="Gender" value={input.gender} />
                <KeyVal label="Race" value={input.race} />
                <KeyVal label="Admission" value={input.admission_type} />
                <KeyVal label="Stay" value={`${input.time_in_hospital}d`} />
                <KeyVal label="Prior Inpatient" value={input.number_inpatient} />
                <KeyVal label="Medications" value={input.num_medications} />
                <KeyVal label="Diagnoses" value={input.number_diagnoses} />
                <KeyVal label="Insulin" value={input.insulin} />
                <KeyVal label="A1C" value={input.a1c_result} />
                <KeyVal label="Discharge" value={input.discharge_group} />
                <KeyVal
                    label="Comorbidities"
                    value={(input.comorbidities as string[]).length || "None"}
                />
            </div>

            {/* Top contributing factors */}
            {result && result.contributing_factors.length > 0 && (
                <div className="flex flex-col gap-1 mt-1">
                    <span className="text-xs font-semibold text-muted-foreground">
                        Top Contributing Factors
                    </span>
                    {result.contributing_factors.slice(0, 5).map((f, i) => (
                        <div
                            key={i}
                            className="flex items-center gap-2 text-xs"
                        >
                            <div className="flex-1 flex items-center gap-1">
                                <span className="font-medium">{f.feature}</span>
                                <span className="text-muted-foreground">
                                    = {f.value}
                                </span>
                            </div>
                            <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full ${f.impact > 0 ? "bg-red-400" : "bg-green-400"}`}
                                    style={{
                                        width: `${Math.min(Math.abs(f.impact) * 200, 100)}%`,
                                    }}
                                />
                            </div>
                            <span
                                className={`w-12 text-right font-mono ${f.impact > 0 ? "text-red-500" : "text-green-500"}`}
                            >
                                {f.impact > 0 ? "+" : ""}
                                {f.impact.toFixed(3)}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

function KeyVal({
    label,
    value,
}: {
    label: string
    value: unknown
}) {
    return (
        <div>
            <span className="text-muted-foreground">{label}: </span>
            <span className="font-medium">{String(value)}</span>
        </div>
    )
}
