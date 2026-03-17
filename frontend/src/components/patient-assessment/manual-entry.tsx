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

export default function ManualEntry({ fields, values, onChange }: ManualEntryProps) {
    const regularFields = fields.filter(f => f.type !== "checkbox_group")
    const checkboxFields = fields.filter(f => f.type === "checkbox_group")

    function renderRegularField(item: any) {
        switch(item["type"]) {
            case "number":
                return (
                    <Field key={item["name"]}>
                        <FieldLabel>{item["label"]}</FieldLabel>
                        <Input
                            id={item["name"]}
                            type="number"
                            value={values[item["name"]] ?? ""}
                            onChange={e => onChange(item["name"], e.target.value)}
                        />
                    </Field>
                )
            case "boolean":
                return (
                    <Field key={item["name"]}>
                        <label className="flex items-center gap-2">
                            <Checkbox
                                checked={!!values[item["name"]]}
                                onCheckedChange={checked => onChange(item["name"], !!checked)}
                            />
                            <FieldLabel className="mb-0">{item["label"]}</FieldLabel>
                        </label>
                    </Field>
                )
            case "select":
                return (
                    <Field key={item["name"]}>
                        <FieldLabel>{item["label"]}</FieldLabel>
                        <Select
                            value={values[item["name"]] ?? ""}
                            onValueChange={val => onChange(item["name"], val)}
                        >
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder={item["label"]} />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectGroup>
                                    {item["options"].map((element: any) => (
                                        <SelectItem key={element["value"]} value={element["value"]}>
                                            {element["label"]}
                                        </SelectItem>
                                    ))}
                                </SelectGroup>
                            </SelectContent>
                        </Select>
                    </Field>
                )
        }
    }

    function renderCheckboxField(item: any) {
        const selected: string[] = values[item["name"]] ?? []
        return (
            <Field key={item["name"]} className="flex flex-col flex-1">
                <FieldLabel>{item["label"]}</FieldLabel>
                <ScrollArea className="flex-1 rounded-md border p-2">
                    <div className="flex flex-col gap-2">
                        {item["options"].map((element: any) => (
                            <label key={element["value"]} className="flex items-center gap-2 text-sm">
                                <Checkbox
                                    checked={selected.includes(element["value"])}
                                    onCheckedChange={(checked) => {
                                        const next = checked
                                            ? [...selected, element["value"]]
                                            : selected.filter((v: string) => v !== element["value"])
                                        onChange(item["name"], next)
                                    }}
                                />
                                {element["label"]}
                            </label>
                        ))}
                    </div>
                </ScrollArea>
            </Field>
        )
    }

    return (
        <div className="flex gap-4 w-full">
            <div className="grid grid-cols-2 gap-4 flex-1 content-start">
                {regularFields.map(renderRegularField)}
            </div>
            {checkboxFields.length > 0 && (
                <div className="flex flex-col gap-4 w-56 self-stretch">
                    {checkboxFields.map(renderCheckboxField)}
                </div>
            )}
        </div>
    )
}
