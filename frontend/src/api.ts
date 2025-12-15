/**
 * API client for Brain Tumor Detection backend
 */

export interface Detection {
  label: string;
  confidence: number;
  bbox: number[];
}

export interface DetectionResponse {
  detections: Detection[];
  image_shape: number[];
  num_detections: number;
}

export interface ImageDetectionResult {
  filename: string;
  detections: Detection[];
  num_detections: number;
  status: string;
  error?: string;
}

export interface BatchProcessingResponse {
  total_images: number;
  successful: number;
  failed: number;
  results: ImageDetectionResult[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
  class_names: string[];
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  return response.json();
}

/**
 * Detect tumors in a single image
 * Returns the annotated image as a blob and detection data
 */
export async function detectSingleImage(
  file: File,
  confidence: number = 0.5
): Promise<{ imageBlob: Blob; detections: Detection[] }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('confidence', confidence.toString());

  const response = await fetch(`${API_BASE_URL}/detect?confidence=${confidence}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Detection failed: ${response.statusText}`);
  }

  // Get detection data from header
  const detectionDataHeader = response.headers.get('X-Detection-Data');
  const detectionData = detectionDataHeader 
    ? JSON.parse(detectionDataHeader) 
    : { detections: [], num_detections: 0 };

  const imageBlob = await response.blob();

  return {
    imageBlob,
    detections: detectionData.detections || [],
  };
}

/**
 * Get detection metadata only (no image)
 */
export async function detectSingleImageJSON(
  file: File,
  confidence: number = 0.5
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/detect-json?confidence=${confidence}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Detection failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Process multiple images and get a ZIP file with results
 */
export async function processBatchImages(
  files: File[],
  confidence: number = 0.5
): Promise<Blob> {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_BASE_URL}/batch?confidence=${confidence}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Batch processing failed: ${response.statusText}`);
  }

  return response.blob();
}

/**
 * Process multiple images and get JSON results (no ZIP)
 */
export async function processBatchImagesJSON(
  files: File[],
  confidence: number = 0.5
): Promise<BatchProcessingResponse> {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_BASE_URL}/batch-json?confidence=${confidence}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Batch processing failed: ${response.statusText}`);
  }

  return response.json();
}

