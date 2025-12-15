import { useState, useEffect } from 'react'
import './App.css'
import SingleImageDetection from './components/SingleImageDetection'
import BatchImageDetection from './components/BatchImageDetection'
import { checkHealth, HealthResponse } from './api'

type Tab = 'single' | 'batch';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('single');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    // Check backend health on mount
    checkHealth()
      .then(setHealth)
      .catch(err => {
        console.error('Health check failed:', err);
        setHealthError(err.message);
      });
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 Brain Tumor Detection System</h1>
        <p className="subtitle">
          Upload MRI images to detect brain tumors using YOLOv10 with CBAM
        </p>
        {health && health.model_loaded && (
          <div className="health-status success">
            ✅ Model loaded successfully
          </div>
        )}
        {healthError && (
          <div className="health-status error">
            ⚠️ Backend connection failed: {healthError}
          </div>
        )}
      </header>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          Single Image Detection
        </button>
        <button
          className={`tab ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          Batch Image Detection
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'single' && <SingleImageDetection />}
        {activeTab === 'batch' && <BatchImageDetection />}
      </div>

      <footer className="app-footer">
        <details>
          <summary>ℹ️ Information</summary>
          <div className="info-content">
            <h3>About the Model</h3>
            <ul>
              <li><strong>Model:</strong> YOLOv10 with CBAM (Convolutional Block Attention Module)</li>
              <li><strong>Classes:</strong> Glioma, Meningioma, No Tumor, Pituitary</li>
              <li><strong>Confidence Threshold:</strong> 50%</li>
            </ul>
            
            <h3>Detection Classes</h3>
            <ul>
              <li><strong>Glioma:</strong> A type of tumor that occurs in the brain and spinal cord</li>
              <li><strong>Meningioma:</strong> A tumor that forms on membranes covering the brain and spinal cord</li>
              <li><strong>No Tumor:</strong> Normal brain tissue without any tumor</li>
              <li><strong>Pituitary:</strong> A tumor in the pituitary gland</li>
            </ul>
          </div>
        </details>
      </footer>
    </div>
  )
}

export default App

