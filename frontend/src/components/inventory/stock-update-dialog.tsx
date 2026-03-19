import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface StockUpdateDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    drugName: string
    drugId: number
    currentStock: number
    onUpdated: () => void
}

export function StockUpdateDialog({ open, onOpenChange, drugName, drugId, currentStock, onUpdated }: StockUpdateDialogProps) {
    const [amount, setAmount] = useState("")
    const [direction, setDirection] = useState<"restock" | "dispense">("restock")
    const [reason, setReason] = useState("restock")
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    function handleSubmit() {
        const qty = parseInt(amount)
        if (!qty || qty <= 0) { setError("Enter a positive number"); return }

        const changeAmount = direction === "restock" ? qty : -qty
        if (direction === "dispense" && qty > currentStock) {
            setError(`Cannot dispense more than current stock (${currentStock})`)
            return
        }

        setLoading(true)
        setError(null)
        fetch(`/api/inventory/${drugId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ change_amount: changeAmount, reason }),
        })
            .then(res => {
                if (!res.ok) return res.json().then(e => { throw new Error(e.detail) })
                return res.json()
            })
            .then(() => {
                onUpdated()
                onOpenChange(false)
                setAmount("")
                setDirection("restock")
                setReason("restock")
            })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }

    return (
        <Sheet open={open} onOpenChange={onOpenChange}>
            <SheetContent>
                <SheetHeader>
                    <SheetTitle>Update Stock</SheetTitle>
                    <SheetDescription>{drugName} — Current: {currentStock}</SheetDescription>
                </SheetHeader>
                <div className="flex flex-col gap-4 p-4">
                    <div className="flex flex-col gap-1.5">
                        <Label>Action</Label>
                        <Select value={direction} onValueChange={v => { setDirection(v as "restock" | "dispense"); setReason(v === "restock" ? "restock" : "dispensed") }}>
                            <SelectTrigger className="w-full">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="restock">Restock</SelectItem>
                                <SelectItem value="dispense">Dispense</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <Label>Quantity</Label>
                        <Input type="number" min={1} value={amount} onChange={e => setAmount(e.target.value)} placeholder="Enter amount" />
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <Label>Reason</Label>
                        <Select value={reason} onValueChange={setReason}>
                            <SelectTrigger className="w-full">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="restock">Restock</SelectItem>
                                <SelectItem value="dispensed">Dispensed</SelectItem>
                                <SelectItem value="expired">Expired</SelectItem>
                                <SelectItem value="adjustment">Adjustment</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    {error && <p className="text-sm text-destructive">{error}</p>}
                    <Button onClick={handleSubmit} disabled={loading}>
                        {loading ? "Updating..." : `${direction === "restock" ? "Restock" : "Dispense"} ${amount || 0} units`}
                    </Button>
                </div>
            </SheetContent>
        </Sheet>
    )
}