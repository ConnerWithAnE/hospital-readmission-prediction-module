import jsPDF from "jspdf"

interface ContributingFactor {
    feature: string
    value: number
    impact: number
}

interface ReportData {
    risk_score: number
    risk_category: string
    contributing_factors: ContributingFactor[]
    patientInput?: Record<string, unknown>
    scenarioName?: string
    scenarioDescription?: string
}

const RISK_RGB: Record<string, [number, number, number]> = {
    very_low: [22, 163, 74],
    low: [34, 197, 94],
    moderate: [234, 179, 8],
    high: [239, 68, 68],
    very_high: [185, 28, 28],
}

const INPUT_LABELS: Record<string, string> = {
    age: "Age",
    gender: "Gender",
    race: "Race",
    time_in_hospital: "Time in Hospital (days)",
    admission_type: "Admission Type",
    admission_source: "Admission Source",
    discharge_group: "Discharge Disposition",
    num_lab_procedures: "Lab Procedures",
    num_procedures: "Procedures",
    num_medications: "Medications",
    number_diagnoses: "Number of Diagnoses",
    number_inpatient: "Prior Inpatient Visits",
    number_outpatient: "Prior Outpatient Visits",
    number_emergency: "Prior Emergency Visits",
    comorbidities: "Comorbidities",
    a1c_result: "A1C Result",
    max_glu_serum: "Max Glucose Serum",
    diabetes_med: "Diabetes Medication",
    med_change: "Medication Change",
    diag_category: "Diagnosis Category",
}

// Medications to show only if active (value !== "No")
const MED_FIELDS = new Set([
    "insulin", "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone", "metformin_rosiglitazone",
    "metformin_pioglitazone",
])

function formatLabel(key: string): string {
    return INPUT_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(val: unknown): string {
    if (val === true) return "Yes"
    if (val === false) return "No"
    if (Array.isArray(val)) return val.length === 0 ? "None" : val.map(v => String(v).replace(/_/g, " ")).join(", ")
    return String(val ?? "N/A").replace(/_/g, " ")
}

export function generatePdfReport(data: ReportData) {
    const doc = new jsPDF({ unit: "mm", format: "letter" })
    const pageW = doc.internal.pageSize.getWidth()
    const margin = 18
    const contentW = pageW - margin * 2
    let y = margin

    // ── Header ──────────────────────────────────────────────────────────────
    doc.setFillColor(30, 41, 59) // slate-800
    doc.rect(0, 0, pageW, 28, "F")

    doc.setTextColor(255, 255, 255)
    doc.setFontSize(18)
    doc.setFont("helvetica", "bold")
    doc.text("Hospital Readmission Risk Assessment", margin, 12)

    doc.setFontSize(9)
    doc.setFont("helvetica", "normal")
    const now = new Date()
    doc.text(`Generated: ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}`, margin, 20)

    if (data.scenarioName) {
        doc.text(`Scenario: ${data.scenarioName}`, pageW - margin, 20, { align: "right" })
    }

    y = 36

    // ── Risk Score Banner ───────────────────────────────────────────────────
    const rgb = RISK_RGB[data.risk_category] ?? [100, 100, 100]
    const percentage = Math.round(data.risk_score * 100)
    const categoryLabel = data.risk_category.replace(/_/g, " ").toUpperCase()

    doc.setFillColor(rgb[0], rgb[1], rgb[2])
    doc.roundedRect(margin, y, contentW, 22, 3, 3, "F")

    doc.setTextColor(255, 255, 255)
    doc.setFontSize(22)
    doc.setFont("helvetica", "bold")
    doc.text(`${categoryLabel} RISK`, margin + 8, y + 10)

    doc.setFontSize(28)
    doc.text(`${percentage}%`, pageW - margin - 8, y + 13, { align: "right" })

    doc.setFontSize(8)
    doc.setFont("helvetica", "normal")
    doc.text("30-day readmission probability", pageW - margin - 8, y + 19, { align: "right" })

    y += 30

    // ── Scenario Description ────────────────────────────────────────────────
    if (data.scenarioDescription) {
        doc.setTextColor(100, 116, 139)
        doc.setFontSize(9)
        doc.setFont("helvetica", "italic")
        doc.text(data.scenarioDescription, margin, y)
        y += 7
    }

    // ── Patient Information ─────────────────────────────────────────────────
    if (data.patientInput && Object.keys(data.patientInput).length > 0) {
        doc.setTextColor(30, 41, 59)
        doc.setFontSize(12)
        doc.setFont("helvetica", "bold")
        doc.text("Patient Information", margin, y)
        y += 2

        doc.setDrawColor(203, 213, 225) // slate-300
        doc.setLineWidth(0.5)
        doc.line(margin, y, pageW - margin, y)
        y += 5

        doc.setFontSize(9)
        const colW = contentW / 2
        let col = 0
        let rowY = y

        // Standard fields first
        const standardEntries: [string, unknown][] = []
        const activeMeds: [string, string][] = []

        for (const [key, val] of Object.entries(data.patientInput)) {
            if (MED_FIELDS.has(key)) {
                if (val && val !== "No") {
                    activeMeds.push([key, String(val)])
                }
            } else {
                standardEntries.push([key, val])
            }
        }

        for (const [key, val] of standardEntries) {
            const xBase = margin + col * colW

            doc.setFont("helvetica", "normal")
            doc.setTextColor(100, 116, 139)
            doc.text(formatLabel(key) + ":", xBase, rowY)

            doc.setFont("helvetica", "bold")
            doc.setTextColor(30, 41, 59)
            const valStr = formatValue(val)
            const truncated = valStr.length > 40 ? valStr.slice(0, 37) + "..." : valStr
            doc.text(truncated, xBase + 45, rowY)

            col++
            if (col >= 2) {
                col = 0
                rowY += 5
            }
        }
        if (col !== 0) rowY += 5

        // Active medications
        if (activeMeds.length > 0) {
            rowY += 2
            doc.setFont("helvetica", "bold")
            doc.setTextColor(100, 116, 139)
            doc.setFontSize(9)
            doc.text("Active Medications:", margin, rowY)
            rowY += 5

            col = 0
            for (const [med, status] of activeMeds) {
                const xBase = margin + col * colW

                doc.setFont("helvetica", "normal")
                doc.setTextColor(100, 116, 139)
                doc.text(formatLabel(med) + ":", xBase, rowY)

                doc.setFont("helvetica", "bold")
                doc.setTextColor(30, 41, 59)
                doc.text(status, xBase + 45, rowY)

                col++
                if (col >= 2) {
                    col = 0
                    rowY += 5
                }
            }
            if (col !== 0) rowY += 5
        }

        y = rowY + 4
    }

    // ── Page break check ────────────────────────────────────────────────────
    const pageH = doc.internal.pageSize.getHeight()
    if (y > pageH - 80) {
        doc.addPage()
        y = margin
    }

    // ── Contributing Factors ────────────────────────────────────────────────
    doc.setTextColor(30, 41, 59)
    doc.setFontSize(12)
    doc.setFont("helvetica", "bold")
    doc.text("Contributing Factors", margin, y)
    y += 2

    doc.setDrawColor(203, 213, 225)
    doc.setLineWidth(0.5)
    doc.line(margin, y, pageW - margin, y)
    y += 5

    // Table header
    doc.setFillColor(241, 245, 249) // slate-100
    doc.rect(margin, y - 3, contentW, 7, "F")
    doc.setFontSize(8)
    doc.setFont("helvetica", "bold")
    doc.setTextColor(71, 85, 105)
    doc.text("Rank", margin + 2, y + 1)
    doc.text("Feature", margin + 15, y + 1)
    doc.text("Value", margin + 85, y + 1)
    doc.text("Impact", margin + 115, y + 1)
    doc.text("Direction", margin + 140, y + 1)
    y += 7

    const factors = data.contributing_factors.slice(0, 10)
    doc.setFontSize(8)

    factors.forEach((f, i) => {
        if (y > pageH - 20) {
            doc.addPage()
            y = margin
        }

        const isPositive = f.impact > 0

        // Alternating row background
        if (i % 2 === 0) {
            doc.setFillColor(248, 250, 252)
            doc.rect(margin, y - 3, contentW, 6, "F")
        }

        doc.setFont("helvetica", "normal")
        doc.setTextColor(30, 41, 59)
        doc.text(`#${i + 1}`, margin + 2, y)

        const featureName = f.feature.replace(/_/g, " ")
        doc.setFont("helvetica", "bold")
        doc.text(featureName.length > 30 ? featureName.slice(0, 27) + "..." : featureName, margin + 15, y)

        doc.setFont("helvetica", "normal")
        doc.text(String(f.value), margin + 85, y)

        // Impact with color
        doc.setTextColor(isPositive ? 239 : 34, isPositive ? 68 : 197, isPositive ? 68 : 94)
        doc.setFont("helvetica", "bold")
        doc.text((isPositive ? "+" : "") + f.impact.toFixed(4), margin + 115, y)

        doc.text(isPositive ? "↑ Risk" : "↓ Risk", margin + 140, y)

        y += 6
    })

    y += 6

    // ── Impact Bar Visual ───────────────────────────────────────────────────
    if (y > pageH - 50) {
        doc.addPage()
        y = margin
    }

    doc.setTextColor(30, 41, 59)
    doc.setFontSize(10)
    doc.setFont("helvetica", "bold")
    doc.text("Impact Distribution", margin, y)
    y += 6

    const maxImpact = Math.max(...factors.map(f => Math.abs(f.impact)))
    const barMaxW = contentW - 60

    factors.slice(0, 8).forEach((f) => {
        if (y > pageH - 15) {
            doc.addPage()
            y = margin
        }

        const isPositive = f.impact > 0
        const barW = (Math.abs(f.impact) / maxImpact) * barMaxW
        const label = f.feature.replace(/_/g, " ")

        doc.setFontSize(7)
        doc.setFont("helvetica", "normal")
        doc.setTextColor(100, 116, 139)
        doc.text(label.length > 25 ? label.slice(0, 22) + "..." : label, margin, y + 1)

        doc.setFillColor(isPositive ? 239 : 34, isPositive ? 68 : 197, isPositive ? 68 : 94)
        doc.roundedRect(margin + 48, y - 2, barW, 4, 1, 1, "F")

        doc.setTextColor(isPositive ? 239 : 34, isPositive ? 68 : 197, isPositive ? 68 : 94)
        doc.setFontSize(7)
        doc.text((isPositive ? "+" : "") + f.impact.toFixed(4), margin + 50 + barW, y + 1)

        y += 6
    })

    y += 8

    // ── Footer / Disclaimer ─────────────────────────────────────────────────
    if (y > pageH - 25) {
        doc.addPage()
        y = margin
    }

    doc.setDrawColor(203, 213, 225)
    doc.setLineWidth(0.3)
    doc.line(margin, y, pageW - margin, y)
    y += 5

    doc.setFontSize(7)
    doc.setFont("helvetica", "italic")
    doc.setTextColor(148, 163, 184)
    const disclaimer = [
        "DISCLAIMER: This report is generated by a machine learning model trained on the UCI Diabetes 130-US Hospitals dataset.",
        "It is intended as a clinical decision support tool only and should not replace professional medical judgment.",
        "Model performance: AUROC 0.672 | Risk predictions reflect statistical patterns and may not capture all clinical nuances.",
    ]
    disclaimer.forEach(line => {
        doc.text(line, margin, y)
        y += 3.5
    })

    // ── Save ────────────────────────────────────────────────────────────────
    const filename = data.scenarioName
        ? `readmission-risk-${data.scenarioName.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.pdf`
        : `readmission-risk-report-${now.toISOString().slice(0, 10)}.pdf`

    doc.save(filename)
}