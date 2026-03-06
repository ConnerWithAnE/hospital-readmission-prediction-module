import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RootLayout } from "./layouts/root-layout";
import { APITester } from "./APITester";

import "src/index.css";

//<Route path="/" element={<Dashboard />} />
//<Route path="/settings" element={<Settings />} />
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<RootLayout />}>
            <Route path="/" element={<APITester />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App
