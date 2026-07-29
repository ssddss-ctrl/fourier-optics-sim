import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Simulator from "./pages/Simulator";
import Simulator2D from "./pages/Simulator2D";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/simulator" element={<Simulator />} />
        <Route path="/simulator-2d" element={<Simulator2D />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
