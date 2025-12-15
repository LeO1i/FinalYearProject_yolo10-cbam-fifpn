import { useState } from 'react';
import { processBatchImages, processBatchImagesJSON, BatchProcessingResponse } from '../api';
import './BatchImageDetection.css';

function BatchImageDetection() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [results, setResults] = useState<BatchProcessingResponse | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

  const handleFilesSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
    setResults(null);
    setZipUrl(null);
    setError(null);
  };

  const handleProcess = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // First, get JSON results for preview
      const jsonResults = await processBatchImagesJSON(selectedFiles, confidence);
      setResults(jsonResults);

      // Then, get ZIP file
      const zipBlob = await processBatchImages(selectedFiles, confidence);
      const url = URL.createObjectURL(zipBlob);
      setZipUrl(url);

    } catch (err) {
      console.error('Batch processing error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process images');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setResults(null);
    setZipUrl(null);
    setError(null);
  };

  return (
    <div className="batch-detection">
      <h2>Batch Image Detection</h2>
      <p className="description">Upload multiple MRI images for batch tumor detection</p>

      <div className="controls">
        <div className="file-input-group">
          <label htmlFor="batch-upload" className="file-label">
            Choose Multiple Images
          </label>
          <input
            id="batch-upload"
            type="file"
            accept="image/*"
            multiple
            onChange={handleFilesSelect}
            disabled={isProcessing}
          />
          {selectedFiles.length > 0 && (
            <span className="file-count">
              {selectedFiles.length} image{selectedFiles.length !== 1 ? 's' : ''} selected
            </span>
          )}
        </div>

        <div className="confidence-slider">
          <label htmlFor="batch-confidence">
            Confidence Threshold: {(confidence * 100).toFixed(0)}%
          </label>
          <input
            id="batch-confidence"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            disabled={isProcessing}
          />
        </div>

        <div className="action-buttons">
          <button
            onClick={handleProcess}
            disabled={selectedFiles.length === 0 || isProcessing}
            className="btn-primary"
          >
            {isProcessing ? '🔄 Processing...' : '🚀 Process All Images'}
          </button>
          <button
            onClick={handleReset}
            disabled={isProcessing || selectedFiles.length === 0}
            className="btn-secondary"
          >
            🔄 Reset
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {selectedFiles.length > 0 && !results && (
        <div className="files-preview">
          <h3>Selected Files:</h3>
          <div className="files-grid">
            {selectedFiles.map((file, idx) => (
              <div key={idx} className="file-card">
                <img src={URL.createObjectURL(file)} alt={file.name} />
                <div className="file-info">
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results && (
        <div className="results-container">
          <div className="summary-card">
            <h3>📊 Processing Summary</h3>
            <div className="summary-stats">
              <div className="stat">
                <span className="stat-label">Total Images:</span>
                <span className="stat-value">{results.total_images}</span>
              </div>
              <div className="stat success">
                <span className="stat-label">Successful:</span>
                <span className="stat-value">{results.successful}</span>
              </div>
              {results.failed > 0 && (
                <div className="stat error">
                  <span className="stat-label">Failed:</span>
                  <span className="stat-value">{results.failed}</span>
                </div>
              )}
            </div>

            {zipUrl && (
              <a href={zipUrl} download="brain_tumor_detection_results.zip" className="download-btn">
                💾 Download All Results (ZIP)
              </a>
            )}
          </div>

          <div className="results-list">
            <h3>Detection Results:</h3>
            {results.results.map((result, idx) => (
              <div key={idx} className={`result-card ${result.status}`}>
                <div className="result-header">
                  <h4>
                    {result.status === 'success' ? '✅' : '❌'} {result.filename}
                  </h4>
                  {result.status === 'success' && (
                    <span className="detection-count">
                      {result.num_detections} detection{result.num_detections !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>

                {result.status === 'success' && result.detections.length > 0 && (
                  <div className="detections-summary">
                    {result.detections.map((detection, dIdx) => (
                      <div key={dIdx} className="detection-item">
                        <span className={`label label-${detection.label.toLowerCase().replace(' ', '-')}`}>
                          {detection.label}
                        </span>
                        <span className="confidence">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {result.status === 'success' && result.detections.length === 0 && (
                  <p className="no-detection">No tumors detected</p>
                )}

                {result.error && (
                  <p className="error-text">Error: {result.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default BatchImageDetection;

