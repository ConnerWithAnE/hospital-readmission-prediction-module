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
    onResult: (result: PredictionResult) => void
}

export default function InfoWindow({ onResult }: InfoWindowProps) {
    const [fields, setFields] = useState<any[]>([])
    const [values, setValues] = useState<Record<string, any>>({})
    const [errors, setErrors] = useState<string[]>([])

    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/fields")
            .then(res => res.json())
            .then(data => setFields(data))
            .catch(err => console.error("Failed to fetch fields:", err))
    }, [])

    function handleChange(name: string, value: any) {
        setValues(prev => ({ ...prev, [name]: value }))
    }

    function handleRunAssessment() {
        const missing = fields
            .filter(f => f.type !== "checkbox_group" && (values[f.name] === undefined || values[f.name] === ""))
            .map(f => f.label)

        if (missing.length > 0) {
            setErrors(missing)
            return
        }

        setErrors([])
        fetch("http://127.0.0.1:8000/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values),
        })
            .then(res => res.json())
            .then(data => onResult(data))
            .catch(err => console.error("Prediction failed:", err))
    }

    return (
        <div className="w-full max-w-[45%] border-2 rounded-lg p-3">
            <Tabs defaultValue="manual">
                <div className="grid grid-cols-2 gap-6">
                    <TabsList className="w-full">
                        <TabsTrigger value="manual" className="col-span-1">Manual Entry</TabsTrigger>
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
                    <Button variant="outline" onClick={handleRunAssessment}>Run Assessment</Button>
                </div>
                {errors.length > 0 && (
                    <p className="text-sm text-destructive mt-2">
                        Missing fields: {errors.join(", ")}
                    </p>
                )}
                <TabsContent value="manual"> 
                    <ManualEntry fields={fields} values={values} onChange={handleChange} />
                </TabsContent>
            </Tabs>
        </div>
    )
}
