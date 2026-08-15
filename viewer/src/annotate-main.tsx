import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { AnnotationApp } from './annotation/AnnotationApp'
import './annotation/annotation.css'

createRoot(document.getElementById('annotation-root')!).render(
  <StrictMode>
    <AnnotationApp />
  </StrictMode>,
)
