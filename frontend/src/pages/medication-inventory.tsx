import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { StockUpdateDialog } from "@/components/inventory/stock-update-dialog"
import { AddDrugDialog } from "@/components/inventory/add-drug-dialog"

interface Drug {
    id: number
    ndc_code: string
    generic_name: string
    brand_name: string | null
    dosage_form: string | null
    strength: string | null
    model_field: string | null
}

interface InventoryItem {
    id: number
    drug: Drug
    quantity_on_hand: number
    reorder_level: number
    unit: string
    last_updated: string | null
    is_low_stock: boolean
}

interface SupplyGap {
    drug: Drug
    quantity_on_hand: number
    reorder_level: number
    model_field: string
    deficit: number
}

export function MedicationInventoryPage() {
    const [items, setItems] = useState<InventoryItem[]>([])
    const [gaps, setGaps] = useState<SupplyGap[]>([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState("")
    const [seeding, setSeeding] = useState(false)
    const [seedResult, setSeedResult] = useState<string | null>(null)

    // Sheet state
    const [addOpen, setAddOpen] = useState(false)
    const [updateTarget, setUpdateTarget] = useState<InventoryItem | null>(null)

    function fetchData() {
        setLoading(true)
        Promise.all([
            fetch("/api/inventory").then(r => r.json()),
            fetch("/api/inventory/supply-gaps").then(r => r.json()),
        ])
            .then(([inv, sg]) => { setItems(inv); setGaps(sg) })
            .catch(console.error)
            .finally(() => setLoading(false))
    }

    useEffect(() => { fetchData() }, [])

    function handleSeed() {
        setSeeding(true)
        setSeedResult(null)
        fetch("/api/drugs/seed", { method: "POST" })
            .then(r => r.json())
            .then(data => {
                setSeedResult(data.message)
                fetchData()
            })
            .catch(e => setSeedResult("Error: " + e.message))
            .finally(() => setSeeding(false))
    }

    function handleRandomize(seed: number) {
        setSeedResult(null)
        fetch(`/api/inventory/randomize?seed=${seed}`, { method: "POST" })
            .then(r => r.json())
            .then(data => {
                setSeedResult(data.detail)
                fetchData()
            })
            .catch(e => setSeedResult("Error: " + e.message))
    }

    function handleDelete(drugId: number, name: string) {
        if (!confirm(`Remove ${name} from inventory?`)) return
        fetch(`/api/inventory/${drugId}`, { method: "DELETE" })
            .then(() => fetchData())
            .catch(console.error)
    }

    const filtered = items.filter(it => {
        if (!search) return true
        const q = search.toLowerCase()
        return (
            it.drug.generic_name.toLowerCase().includes(q) ||
            (it.drug.brand_name?.toLowerCase().includes(q) ?? false) ||
            it.drug.ndc_code.toLowerCase().includes(q) ||
            (it.drug.model_field?.toLowerCase().includes(q) ?? false)
        )
    })

    const totalTracked = items.length
    const lowStockCount = items.filter(i => i.is_low_stock && i.quantity_on_hand > 0).length
    const zeroStockCount = items.filter(i => i.quantity_on_hand === 0).length

    return (
        <div className="flex flex-col gap-6 p-6 w-full h-full overflow-y-auto">
            <h1 className="text-2xl font-bold">Medication Inventory</h1>

            {/* Supply Gap Alerts */}
            {gaps.length > 0 && (
                <div className="rounded-lg border-2 border-red-300 bg-red-50 p-4">
                    <h2 className="text-sm font-semibold text-red-700 mb-2">
                        Supply Gap Alert — {gaps.length} model-linked medication{gaps.length > 1 ? "s" : ""} low or out of stock
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                        {gaps.map(g => (
                            <div
                                key={g.drug.id}
                                className={`rounded-md border p-2 text-xs ${
                                    g.quantity_on_hand === 0
                                        ? "border-red-400 bg-red-100 text-red-800"
                                        : "border-amber-400 bg-amber-50 text-amber-800"
                                }`}
                            >
                                <span className="font-semibold">{g.drug.generic_name}</span>
                                <span className="text-muted-foreground ml-1">({g.model_field})</span>
                                <div className="mt-0.5">
                                    Stock: {g.quantity_on_hand} / Reorder: {g.reorder_level}
                                    <span className="ml-1 font-semibold">
                                        {g.quantity_on_hand === 0 ? "OUT OF STOCK" : `Deficit: ${g.deficit}`}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Summary Cards */}
            <div className="grid grid-cols-3 gap-4">
                <SummaryCard label="Total Tracked" value={totalTracked} color="text-foreground" />
                <SummaryCard label="Low Stock" value={lowStockCount} color={lowStockCount > 0 ? "text-amber-600" : "text-foreground"} />
                <SummaryCard label="Out of Stock" value={zeroStockCount} color={zeroStockCount > 0 ? "text-red-600" : "text-foreground"} />
            </div>

            {/* Action Bar */}
            <div className="flex items-center gap-3 flex-wrap">
                <Input
                    className="max-w-xs"
                    placeholder="Search by name, NDC, or model field..."
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                />
                <div className="flex-1" />
                <Button variant="outline" size="sm" onClick={() => setAddOpen(true)}>
                    + Add Drug
                </Button>
                <Button variant="outline" size="sm" onClick={handleSeed} disabled={seeding}>
                    {seeding ? "Seeding..." : "Seed from NDC"}
                </Button>
                <div className="border-l pl-3 flex items-center gap-1.5">
                    <span className="text-xs text-muted-foreground">Test:</span>
                    <Button variant="outline" size="xs" onClick={() => handleRandomize(1)}>Preset 1</Button>
                    <Button variant="outline" size="xs" onClick={() => handleRandomize(2)}>Preset 2</Button>
                    <Button variant="outline" size="xs" onClick={() => handleRandomize(3)}>Preset 3</Button>
                    <Button variant="outline" size="xs" onClick={() => handleRandomize(0)}>Random</Button>
                </div>
            </div>

            {seedResult && (
                <p className="text-sm text-muted-foreground bg-muted rounded-md px-3 py-2">{seedResult}</p>
            )}

            {/* Inventory Table */}
            {loading ? (
                <p className="text-sm text-muted-foreground">Loading inventory...</p>
            ) : filtered.length === 0 ? (
                <div className="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground border-2 rounded-lg">
                    <p className="text-sm">{items.length === 0 ? "No drugs in inventory yet." : "No results match your search."}</p>
                    {items.length === 0 && (
                        <p className="text-xs">Click "Seed from NDC" to populate with diabetes medications, or "Add Drug" to add manually.</p>
                    )}
                </div>
            ) : (
                <div className="border-2 rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="bg-muted/50 border-b">
                                <th className="text-left p-3 font-semibold">NDC Code</th>
                                <th className="text-left p-3 font-semibold">Generic Name</th>
                                <th className="text-left p-3 font-semibold">Brand</th>
                                <th className="text-left p-3 font-semibold">Form / Strength</th>
                                <th className="text-left p-3 font-semibold">Model Field</th>
                                <th className="text-center p-3 font-semibold">Stock</th>
                                <th className="text-center p-3 font-semibold">Reorder</th>
                                <th className="text-center p-3 font-semibold">Unit</th>
                                <th className="text-center p-3 font-semibold">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map(it => {
                                const stockColor = it.quantity_on_hand === 0
                                    ? "text-red-600 font-bold"
                                    : it.is_low_stock
                                        ? "text-amber-600 font-semibold"
                                        : "text-green-600"

                                return (
                                    <tr key={it.drug.id} className="border-b last:border-b-0 hover:bg-muted/30">
                                        <td className="p-3 font-mono text-xs">{it.drug.ndc_code}</td>
                                        <td className="p-3 font-medium">{it.drug.generic_name}</td>
                                        <td className="p-3 text-muted-foreground">{it.drug.brand_name ?? "—"}</td>
                                        <td className="p-3 text-muted-foreground">
                                            {[it.drug.dosage_form, it.drug.strength].filter(Boolean).join(" ") || "—"}
                                        </td>
                                        <td className="p-3">
                                            {it.drug.model_field ? (
                                                <span className="inline-block rounded-full bg-blue-100 text-blue-700 px-2 py-0.5 text-xs font-medium">
                                                    {it.drug.model_field}
                                                </span>
                                            ) : (
                                                <span className="text-muted-foreground text-xs">—</span>
                                            )}
                                        </td>
                                        <td className={`p-3 text-center ${stockColor}`}>{it.quantity_on_hand}</td>
                                        <td className="p-3 text-center text-muted-foreground">{it.reorder_level}</td>
                                        <td className="p-3 text-center text-muted-foreground">{it.unit}</td>
                                        <td className="p-3 text-center">
                                            <div className="flex items-center justify-center gap-1">
                                                <Button
                                                    variant="outline"
                                                    size="xs"
                                                    onClick={() => setUpdateTarget(it)}
                                                >
                                                    Update
                                                </Button>
                                                <Button
                                                    variant="ghost"
                                                    size="xs"
                                                    className="text-destructive hover:text-destructive"
                                                    onClick={() => handleDelete(it.drug.id, it.drug.generic_name)}
                                                >
                                                    Remove
                                                </Button>
                                            </div>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Dialogs */}
            <AddDrugDialog open={addOpen} onOpenChange={setAddOpen} onAdded={fetchData} />
            {updateTarget && (
                <StockUpdateDialog
                    open={!!updateTarget}
                    onOpenChange={open => { if (!open) setUpdateTarget(null) }}
                    drugName={updateTarget.drug.generic_name}
                    drugId={updateTarget.drug.id}
                    currentStock={updateTarget.quantity_on_hand}
                    onUpdated={fetchData}
                />
            )}
        </div>
    )
}

function SummaryCard({ label, value, color }: { label: string; value: number; color: string }) {
    return (
        <div className="rounded-lg border-2 p-4 flex flex-col items-center">
            <span className={`text-3xl font-bold ${color}`}>{value}</span>
            <span className="text-xs text-muted-foreground">{label}</span>
        </div>
    )
}