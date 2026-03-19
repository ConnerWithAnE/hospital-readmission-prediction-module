import { useEffect, useState } from "react"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip"
import { Button } from "@/components/ui/button"
import ManualEntry from "./manual-entry"
import type { PredictionResult } from "@/pages/patient-assesment"

interface InfoWindowProps {
    onResult: (result: PredictionResult, values: Record<string, any>) => void
    onClear?: () => void
}

const STORAGE_KEY = "patient-assessment-values"

function loadSavedValues(): Record<string, any> {
    try {
        const saved = sessionStorage.getItem(STORAGE_KEY)
        return saved ? JSON.parse(saved) : {}
    } catch { return {} }
}

export default function InfoWindow({ onResult, onClear }: InfoWindowProps) {
    const [fields, setFields] = useState<any[]>([])
    const [values, setValues] = useState<Record<string, any>>(loadSavedValues)
    const [errors, setErrors] = useState<string[]>([])
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        fetch("/api/fields")
            .then(res => res.json())
            .then(data => setFields(data))
            .catch(err => console.error("Failed to fetch fields:", err))
    }, [])

    function handleChange(name: string, value: any) {
        setValues(prev => {
            const next = { ...prev, [name]: value }
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next))
            return next
        })
    }

    function handleClear() {
        setValues({})
        setErrors([])
        sessionStorage.removeItem(STORAGE_KEY)
        onClear?.()
    }

    function handleRunAssessment() {
        const missing = fields
            .filter(f => f.type !== "checkbox_group" && f.type !== "boolean" && f.type !== "med_status_group" && (values[f.name] === undefined || values[f.name] === ""))
            .map(f => f.label)

        if (missing.length > 0) {
            setErrors(missing)
            return
        }

        setErrors([])
        setLoading(true)
        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values),
        })
            .then(res => res.json())
            .then(data => onResult(data, values))
            .catch(err => console.error("Prediction failed:", err))
            .finally(() => setLoading(false))
    }

    return (
        <div className="w-full max-w-[45%] border-2 rounded-lg flex flex-col">
            <Tabs defaultValue="manual" className="flex flex-col flex-1">
                <div className="flex items-center gap-3 p-3 border-b">
                    <TabsList>
                        <TabsTrigger value="manual">Manual Entry</TabsTrigger>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <span>
                                        <TabsTrigger value="upload" disabled>Upload</TabsTrigger>
                                    </span>
                                </TooltipTrigger>
                                <TooltipContent>Coming soon</TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </TabsList>
                    <div className="flex-1" />
                    <Button variant="outline" size="sm" onClick={handleClear}>
                        Clear
                    </Button>
                    <Button onClick={handleRunAssessment} disabled={loading}>
                        {loading ? "Running..." : "Run Assessment"}
                    </Button>
                </div>
                {errors.length > 0 && (
                    <div className="px-3 py-2 bg-destructive/10 border-b">
                        <p className="text-sm text-destructive">
                            Missing fields: {errors.join(", ")}
                        </p>
                    </div>
                )}
                <TabsContent value="manual" className="flex-1 p-3 mt-0">
                    <ManualEntry fields={fields} values={values} onChange={handleChange} />
                </TabsContent>
            </Tabs>
        </div>
    )
}