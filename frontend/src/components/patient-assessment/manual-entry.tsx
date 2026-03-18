import { Field, FieldLabel } from "@/components/ui/field"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Checkbox } from "../ui/checkbox"

interface ManualEntryProps {
    fields: any[]
    values: Record<string, any>
    onChange: (name: string, value: any) => void
}

const FIELD_SECTIONS: { label: string; fields: string[] }[] = [
    {
        label: "Demographics",
        fields: ["age", "gender", "race"],
    },
    {
        label: "Admission Details",
        fields: ["admission_type", "admission_source", "discharge_group", "time_in_hospital"],
    },
    {
        label: "Clinical History",
        fields: [
            "number_diagnoses", "diag_category",
            "num_lab_procedures", "num_procedures",
            "number_inpatient", "number_outpatient", "number_emergency",
        ],
    },
    {
        label: "Medications",
        fields: [
            "num_medications", "n_active_meds", "n_med_changes",
            "insulin", "metformin",
            "diabetes_med", "med_change",
        ],
    },
    {
        label: "Lab Results",
        fields: ["a1c_result", "max_glu_serum"],
    },
]

export default function ManualEntry({ fields, values, onChange }: ManualEntryProps) {
    const checkboxFields = fields.filter(f => f.type === "checkbox_group")
    const fieldMap = new Map(fields.map(f => [f.name, f]))

    // Collect any fields not in a section (safety net)
    const assigned = new Set(FIELD_SECTIONS.flatMap(s => s.fields))
    const unassigned = fields.filter(f => f.type !== "checkbox_group" && !assigned.has(f.name))

    return (
        <ScrollArea className="h-[calc(100vh-12rem)]">
            <div className="flex flex-col gap-5 pr-4">
                {FIELD_SECTIONS.map(section => {
                    const sectionFields = section.fields
                        .map(name => fieldMap.get(name))
                        .filter(Boolean)
                    if (sectionFields.length === 0) return null

                    return (
                        <Section key={section.label} label={section.label}>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                                {sectionFields.map(item => (
                                    <FieldRenderer
                                        key={item.name}
                                        item={item}
                                        value={values[item.name]}
                                        onChange={onChange}
                                    />
                                ))}
                            </div>
                        </Section>
                    )
                })}

                {unassigned.length > 0 && (
                    <Section label="Other">
                        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                            {unassigned.map(item => (
                                <FieldRenderer
                                    key={item.name}
                                    item={item}
                                    value={values[item.name]}
                                    onChange={onChange}
                                />
                            ))}
                        </div>
                    </Section>
                )}

                {checkboxFields.length > 0 &&
                    checkboxFields.map(item => (
                        <Section key={item.name} label={item.label}>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                                {item.options.map((opt: any) => {
                                    const selected: string[] = values[item.name] ?? []
                                    return (
                                        <label
                                            key={opt.value}
                                            className="flex items-center gap-2 text-sm py-0.5 cursor-pointer hover:text-foreground text-muted-foreground has-[:checked]:text-foreground"
                                        >
                                            <Checkbox
                                                checked={selected.includes(opt.value)}
                                                onCheckedChange={(checked) => {
                                                    const next = checked
                                                        ? [...selected, opt.value]
                                                        : selected.filter((v: string) => v !== opt.value)
                                                    onChange(item.name, next)
                                                }}
                                            />
                                            {opt.label}
                                        </label>
                                    )
                                })}
                            </div>
                        </Section>
                    ))}
            </div>
        </ScrollArea>
    )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="flex flex-col gap-2.5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b pb-1.5">
                {label}
            </h3>
            {children}
        </div>
    )
}

function FieldRenderer({
    item,
    value,
    onChange,
}: {
    item: any
    value: any
    onChange: (name: string, value: any) => void
}) {
    switch (item.type) {
        case "number":
            return (
                <Field>
                    <FieldLabel className="text-xs">{item.label}</FieldLabel>
                    <Input
                        id={item.name}
                        type="number"
                        min={item.min}
                        max={item.max}
                        value={value ?? ""}
                        onChange={e => onChange(item.name, e.target.value)}
                        className="h-8"
                    />
                </Field>
            )
        case "boolean":
            return (
                <Field>
                    <label className="flex items-center gap-2 h-8 cursor-pointer">
                        <Checkbox
                            checked={!!value}
                            onCheckedChange={checked => onChange(item.name, !!checked)}
                        />
                        <span className="text-sm">{item.label}</span>
                    </label>
                </Field>
            )
        case "select":
            return (
                <Field>
                    <FieldLabel className="text-xs">{item.label}</FieldLabel>
                    <Select
                        value={value ?? ""}
                        onValueChange={val => onChange(item.name, val)}
                    >
                        <SelectTrigger className="w-full h-8">
                            <SelectValue placeholder="Select..." />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectGroup>
                                {item.options.map((opt: any) => (
                                    <SelectItem key={opt.value} value={opt.value}>
                                        {opt.label}
                                    </SelectItem>
                                ))}
                            </SelectGroup>
                        </SelectContent>
                    </Select>
                </Field>
            )
        default:
            return null
    }
}