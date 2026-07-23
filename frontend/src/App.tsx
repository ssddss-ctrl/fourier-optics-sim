import { BrowserRouter, Routes, Route } from 'react-router-dom'

function Hello() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <h1 className="text-3xl font-semibold">Hello</h1>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Hello />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
