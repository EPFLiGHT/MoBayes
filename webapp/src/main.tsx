import { ChakraProvider, defaultSystem } from '@chakra-ui/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import App from './App'
import DebugChat from './routes/DebugChat'
import DebugLayout from './routes/DebugLayout'
import DebugParse from './routes/DebugParse'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ChakraProvider value={defaultSystem}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/debug" element={<DebugLayout />}>
            <Route index element={<Navigate to="parse" replace />} />
            <Route path="parse" element={<DebugParse />} />
            <Route path="chat" element={<DebugChat />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ChakraProvider>
  </StrictMode>,
)
