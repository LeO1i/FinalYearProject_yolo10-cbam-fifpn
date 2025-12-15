import { useState } from 'react';
import { detectSingleImage, Detection } from '../api';
import './SingleImageDetection.css';

function SingleImageDetection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.5);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultUrl(null);
      setDetections([]);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const result = await detectSingleImage(selectedFile, confidence);
      
      // Create URL for the result image
      const url = URL.createObjectURL(result.imageBlob);
      setResultUrl(url);
      setDetections(result.detections);
      
    } catch (err) {
      console.error('Detection error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResultUrl(null);
    setDetections([]);
    setError(null);
  };

  return (
    <div className="single-detection">
      <h2>Single Image Detection</h2>
      <p className="description">Upload a single MRI image for tumor detection</p>

      <div className="controls">
        <div className="file-input-group">
          <label htmlFor="image-upload" className="file-label">
            Choose Image
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            disabled={isProcessing}
          />
          {selectedFile && (
            <span className="file-name">{selectedFile.name}</span>
          )}
        </div>

        <div className="confidence-slider">
          <label htmlFor="confidence">
            Confidence Threshold: {(confidence * 100).toFixed(0)}%
          </label>
          <input
            id="confidence"
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
            onClick={handleDetect}
            disabled={!selectedFile || isProcessing}
            className="btn-primary"
          >
            {isProcessing ? '🔄 Processing...' : '🔍 Detect Tumor'}
          </button>
          <button
            onClick={handleReset}
            disabled={isProcessing || !selectedFile}
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

      <div className="results-container">
        <div className="image-comparison">
          {previewUrl && (
            <div className="image-box">
              <h3>Original Image</h3>
              <img src={previewUrl} alt="Original" />
            </div>
          )}

          {resultUrl && (
            <div className="image-box">
              <h3>Detection Result</h3>
              <img src={resultUrl} alt="Detection Result" />
              <a
                href={resultUrl}
                download={`detected_${selectedFile?.name || 'result.jpg'}`}
                className="download-link"
              >
                💾 Download Result
              </a>
            </div>
          )}
        </div>

        {detections.length > 0 && (
          <div className="detections-list">
            <h3>Detections Found: {detections.length}</h3>
            <table className="detections-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Class</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((detection, idx) => (
                  <tr key={idx}>
                    <td>{idx + 1}</td>
                    <td>
                      <span className={`label label-${detection.label.toLowerCase().replace(' ', '-')}`}>
                        {detection.label}
                      </span>
                    </td>
                    <td>{(detection.confidence * 100).toFixed(1)}%</td>
                    <td className="bbox-coords">
                      [{detection.bbox.map(v => v.toFixed(0)).join(', ')}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {resultUrl && detections.length === 0 && (
          <div className="no-detections">
            ℹ️ No tumors detected in this image
          </div>
        )}
      </div>
    </div>
  );
}

export default SingleImageDetection;

