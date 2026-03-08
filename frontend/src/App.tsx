import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RootLayout } from "./layouts/root-layout";
import { APITester } from "./APITester";

import "src/index.css";
import { PatientAssessmentPage } from "./pages/patient-assesment";

//<Route path="/" element={<Dashboard />} />
//<Route path="/settings" element={<Settings />} />
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<RootLayout />}>
            <Route path="/" element={<PatientAssessmentPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App
