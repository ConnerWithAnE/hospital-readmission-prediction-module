import { useState, useEffect } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface AddDrugDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    onAdded: () => void
}

const NONE_VALUE = "__none__"

export function AddDrugDialog({ open, onOpenChange, onAdded }: AddDrugDialogProps) {
    const [modelMeds, setModelMeds] = useState<string[]>([])
    const [form, setForm] = useState({
        ndc_code: "",
        generic_name: "",
        brand_name: "",
        dosage_form: "",
        strength: "",
        model_field: "",
        initial_quantity: "0",
        reorder_level: "10",
        unit: "units",
    })
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetch("/api/inventory/model-medications")
            .then(r => r.json())
            .then(setModelMeds)
            .catch(() => {})
    }, [])

    function update(key: string, value: string) {
        setForm(prev => ({ ...prev, [key]: value }))
    }

    function handleSubmit() {
        if (!form.ndc_code.trim() || !form.generic_name.trim()) {
            setError("NDC code and generic name are required")
            return
        }
        setLoading(true)
        setError(null)

        const body = {
            ...form,
            model_field: form.model_field === NONE_VALUE ? null : (form.model_field || null),
            initial_quantity: parseInt(form.initial_quantity) || 0,
            reorder_level: parseInt(form.reorder_level) || 10,
        }

        fetch("/api/inventory", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })
            .then(res => {
                if (!res.ok) return res.json().then(e => { throw new Error(e.detail) })
                return res.json()
            })
            .then(() => {
                onAdded()
                onOpenChange(false)
                setForm({ ndc_code: "", generic_name: "", brand_name: "", dosage_form: "", strength: "", model_field: "", initial_quantity: "0", reorder_level: "10", unit: "units" })
            })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }

    return (
        <Sheet open={open} onOpenChange={onOpenChange}>
            <SheetContent>
                <SheetHeader>
                    <SheetTitle>Add Drug to Inventory</SheetTitle>
                    <SheetDescription>Add a new medication to track</SheetDescription>
                </SheetHeader>
                <div className="flex flex-col gap-3 p-4 overflow-y-auto">
                    <div className="flex flex-col gap-1.5">
                        <Label>NDC Code *</Label>
                        <Input value={form.ndc_code} onChange={e => update("ndc_code", e.target.value)} placeholder="e.g. 0002-7510-01" />
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <Label>Generic Name *</Label>
                        <Input value={form.generic_name} onChange={e => update("generic_name", e.target.value)} placeholder="e.g. METFORMIN HYDROCHLORIDE" />
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <Label>Brand Name</Label>
                        <Input value={form.brand_name} onChange={e => update("brand_name", e.target.value)} placeholder="e.g. Glucophage" />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="flex flex-col gap-1.5">
                            <Label>Dosage Form</Label>
                            <Input value={form.dosage_form} onChange={e => update("dosage_form", e.target.value)} placeholder="e.g. TABLET" />
                        </div>
                        <div className="flex flex-col gap-1.5">
                            <Label>Strength</Label>
                            <Input value={form.strength} onChange={e => update("strength", e.target.value)} placeholder="e.g. 500mg" />
                        </div>
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <Label>Model Medication Field</Label>
                        <Select value={form.model_field || NONE_VALUE} onValueChange={v => update("model_field", v === NONE_VALUE ? "" : v)}>
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder="Not mapped to model" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value={NONE_VALUE}>Not mapped</SelectItem>
                                {modelMeds.map(m => (
                                    <SelectItem key={m} value={m}>{m}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">Maps this drug to a prediction model medication field</p>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="flex flex-col gap-1.5">
                            <Label>Initial Qty</Label>
                            <Input type="number" min={0} value={form.initial_quantity} onChange={e => update("initial_quantity", e.target.value)} />
                        </div>
                        <div className="flex flex-col gap-1.5">
                            <Label>Reorder Level</Label>
                            <Input type="number" min={0} value={form.reorder_level} onChange={e => update("reorder_level", e.target.value)} />
                        </div>
                        <div className="flex flex-col gap-1.5">
                            <Label>Unit</Label>
                            <Select value={form.unit} onValueChange={v => update("unit", v)}>
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="units">Units</SelectItem>
                                    <SelectItem value="tablets">Tablets</SelectItem>
                                    <SelectItem value="vials">Vials</SelectItem>
                                    <SelectItem value="bottles">Bottles</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                    {error && <p className="text-sm text-destructive">{error}</p>}
                    <Button onClick={handleSubmit} disabled={loading}>
                        {loading ? "Adding..." : "Add Drug"}
                    </Button>
                </div>
            </SheetContent>
        </Sheet>
    )
}