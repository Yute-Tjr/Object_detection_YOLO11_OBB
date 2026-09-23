export type TaskStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "partial_failed"
  | "failed";

export type ImageStatus = "queued" | "running" | "succeeded" | "failed";

export type ImageStage =
  | "pending"
  | "object_detection"
  | "anomaly_classification"
  | "rendering"
  | "complete";

export type OverallResult = "OK" | "NG" | "UNKNOWN";

export interface AuthUser {
  id: string;
  username: string;
}

export interface HealthResponse {
  apiReady: boolean;
  databaseReady: boolean;
  workerReady: boolean;
  modelsReady: boolean;
}

export interface ClassificationResult {
  classifierType: "anomaly" | "color" | string;
  predictedLabel: string;
  confidence: number;
  probabilities?: Record<string, number> | null;
}

export interface DetectionResult {
  id: string;
  regionLabel: string;
  confidence: number;
  points: number[][];
  selectedForClassification: boolean;
  cropUrl?: string | null;
  anomaly?: ClassificationResult | null;
  color?: ClassificationResult | null;
}

export interface ImageSummary {
  id: string;
  sequenceNo: number;
  originalFilename: string;
  status: ImageStatus;
  stage: ImageStage;
  overallResult: OverallResult;
  width: number;
  height: number;
  sizeBytes: number;
  originalUrl: string;
  resultUrl?: string | null;
  errorCode?: string | null;
  errorMessage?: string | null;
}

export interface ImageDetail extends ImageSummary {
  detections: DetectionResult[];
}

export interface TaskSummary {
  id: string;
  displayId: string;
  status: TaskStatus;
  currentStage: ImageStage;
  totalImages: number;
  completedImages: number;
  succeededImages: number;
  failedImages: number;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface TaskDetail extends TaskSummary {
  images: ImageSummary[];
}

export interface TaskPage {
  items: TaskSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface ImagePage {
  items: ImageSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface ListTaskParams {
  status?: "ongoing" | "all" | "success" | "failed" | TaskStatus;
  query?: string;
  offset?: number;
  limit?: number;
}
