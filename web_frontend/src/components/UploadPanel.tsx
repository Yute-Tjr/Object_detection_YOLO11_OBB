import {
  FolderOpen,
  Play,
  Trash,
  UploadSimple,
  X,
} from "@phosphor-icons/react";
import { useRef, useState } from "react";

import type { CreateTaskMetadata, HealthResponse } from "../api/types";


interface UploadPanelProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  metadata: CreateTaskMetadata;
  onMetadataChange: (metadata: CreateTaskMetadata) => void;
  onStart: () => void;
  health: HealthResponse | null;
  loadingHealth: boolean;
  submitting: boolean;
}


export function UploadPanel({
  files,
  onFilesChange,
  metadata,
  onMetadataChange,
  onStart,
  health,
  loadingHealth,
  submitting,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const tooMany = files.length > 100;
  const unavailable = health?.models.filter((model) => !model.ready) ?? [];
  const detectorName = health?.models.find((model) => model.modelType === "detector")?.name
    ?? "YOLO11l-OBB";
  const ready = Boolean(
    health?.apiReady && health.databaseReady && health.workerReady && health.modelsReady,
  );
  const disabled = files.length === 0
    || tooMany
    || !metadata.operator.trim()
    || !ready
    || submitting;

  const appendFiles = (incoming: FileList | File[]) => {
    onFilesChange([...files, ...Array.from(incoming)]);
  };

  return (
    <section
      className={dragging ? "upload-panel is-dragging" : "upload-panel"}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        appendFiles(event.dataTransfer.files);
      }}
      aria-label="图片上传"
    >
      <div className="task-metadata-form" aria-label="任务信息">
        <label>
          <span>操作员 <em>必填</em></span>
          <input
            aria-label="操作员"
            autoComplete="off"
            maxLength={128}
            required
            value={metadata.operator}
            onChange={(event) => onMetadataChange({
              ...metadata,
              operator: event.target.value,
            })}
            placeholder="请输入操作员姓名"
          />
        </label>
        <label>
          <span>任务名称</span>
          <input
            aria-label="任务名称"
            maxLength={255}
            value={metadata.name ?? ""}
            onChange={(event) => onMetadataChange({
              ...metadata,
              name: event.target.value,
            })}
            placeholder="例如：端子_产线A_早班"
          />
        </label>
        <label className="task-metadata-form__note">
          <span>备注</span>
          <textarea
            aria-label="备注"
            rows={2}
            value={metadata.note ?? ""}
            onChange={(event) => onMetadataChange({
              ...metadata,
              note: event.target.value,
            })}
            placeholder="选填"
          />
        </label>
      </div>

      <div className="upload-panel__prompt">
        <UploadSimple size={38} weight="regular" />
        <div>
          <strong>拖拽图片到此处，或点击选择图片</strong>
          <p>支持 JPG、JPEG、PNG、BMP 格式，单次最多 100 张图片</p>
        </div>
      </div>

      <div className="upload-panel__actions">
        <div className="upload-panel__count">
          <span className="sr-only">已选择 {files.length} 张图片</span>
          已选择 <strong>{files.length}</strong> 张图片
          <small>（上限 100 张）</small>
          <small>当前模型 <span>{detectorName}</span></small>
        </div>
        <input
          ref={inputRef}
          className="sr-only"
          id="image-picker"
          aria-label="选择图片"
          type="file"
          accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
          multiple
          onChange={(event) => {
            if (event.target.files) appendFiles(event.target.files);
            event.target.value = "";
          }}
        />
        <button type="button" className="button button--secondary" onClick={() => inputRef.current?.click()}>
          <FolderOpen size={20} />
          选择图片
        </button>
        <button type="button" className="button button--primary" disabled={disabled} onClick={onStart}>
          <Play size={19} weight="fill" />
          {submitting ? "正在创建" : "开始检测"}
        </button>
      </div>

      {(files.length > 0 || tooMany || unavailable.length > 0) && (
        <div className="upload-panel__footer">
          <div className="file-chips" aria-label="已选图片">
            {files.slice(0, 5).map((file, index) => (
              <span className="file-chip" key={`${file.name}-${file.lastModified}-${index}`}>
                {file.name}
                <button
                  aria-label={`移除 ${file.name}`}
                  onClick={() => onFilesChange(files.filter((_, itemIndex) => itemIndex !== index))}
                >
                  <X size={14} />
                </button>
              </span>
            ))}
            {files.length > 5 && <span className="file-chip file-chip--more">+{files.length - 5}</span>}
          </div>
          {files.length > 0 && (
            <button className="text-button" aria-label="清空已选图片" onClick={() => onFilesChange([])}>
              <Trash size={16} /> 清空
            </button>
          )}
          {tooMany && <p className="inline-error">单次最多 100 张图片</p>}
          {!loadingHealth && unavailable.map((model) => (
            <p className="inline-error" key={model.name}>{model.name} 尚未就绪</p>
          ))}
          {!loadingHealth && health && !health.workerReady && unavailable.length === 0 && (
            <p className="inline-error">推理 Worker 尚未就绪</p>
          )}
        </div>
      )}
    </section>
  );
}
