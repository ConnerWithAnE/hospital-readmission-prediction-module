import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RootLayout } from "./layouts/root-layout";
import { APITester } from "./APITester";

import "src/index.css";
import { PatientAssessmentPage } from "./pages/patient-assesment";
import { ModelAssessmentPage } from "./pages/model-assessment";
import { ScenarioComparisonPage } from "./pages/scenario-comparison";
import { MedicationInventoryPage } from "./pages/medication-inventory";

//<Route path="/" element={<Dashboard />} />
//<Route path="/settings" element={<Settings />} />
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<RootLayout />}>
            <Route path="/" element={<PatientAssessmentPage />} />
            <Route path="/model_stats" element={<ModelAssessmentPage />} />
            <Route path="/scenarios" element={<ScenarioComparisonPage />} />
            <Route path="/inventory" element={<MedicationInventoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App
