import { WarningCircle, X } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";

import { apiClient, type ApiClient } from "../api/client";
import type {
  FeedbackColor,
  FeedbackUpdateRequest,
  FeedbackVerdict,
  ImageFeedbackView,
  LogicalRegion,
} from "../api/types";


export type FeedbackClient = Pick<ApiClient, "getImageFeedback" | "updateImageFeedback">;

interface FeedbackDialogProps {
  imageId: string;
  filename: string;
  client?: FeedbackClient;
  onClose: () => void;
  onSaved: (feedback: ImageFeedbackView) => void;
}

interface DraftItem {
  verdict?: FeedbackVerdict;
  color?: FeedbackColor;
}

interface FeedbackDraft {
  items: Record<string, DraftItem>;
  missedRegions: Set<LogicalRegion>;
}

const COLORS: FeedbackColor[] = ["B", "G", "R", "W"];


function draftFromView(view: ImageFeedbackView): FeedbackDraft {
  const existing = new Map(
    (view.feedback?.items ?? []).map((item) => [item.detectionId, item]),
  );
  const items: Record<string, DraftItem> = {};
  view.detections.forEach((detection) => {
    const feedback = existing.get(detection.detectionId);
    items[detection.detectionId] = feedback ? {
      verdict: feedback.verdict,
      color: feedback.color ?? undefined,
    } : {};
  });
  return {
    items,
    missedRegions: new Set(view.feedback?.missedRegions ?? []),
  };
}


export function FeedbackDialog({
  imageId,
  filename,
  client = apiClient,
  onClose,
  onSaved,
}: FeedbackDialogProps) {
  const [view, setView] = useState<ImageFeedbackView | null>(null);
  const [draft, setDraft] = useState<FeedbackDraft | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const closeTimer = useRef<number | null>(null);

  const load = useCallback((signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    return client.getImageFeedback(imageId, signal).then((loaded) => {
      setView(loaded);
      setDraft(draftFromView(loaded));
    }).catch((caught: unknown) => {
      if (signal?.aborted) return;
      setError(caught instanceof Error ? caught.message : "反馈加载失败");
    }).finally(() => {
      if (!signal?.aborted) setLoading(false);
    });
  }, [client, imageId]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !submitting) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, submitting]);

  useEffect(() => () => {
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
  }, []);

  const setItem = (detectionId: string, update: Partial<DraftItem>) => {
    setDraft((current) => current ? {
      ...current,
      items: {
        ...current.items,
        [detectionId]: { ...current.items[detectionId], ...update },
      },
    } : current);
    setError(null);
  };

  const toggleMissed = (region: LogicalRegion) => {
    setDraft((current) => {
      if (!current) return current;
      const missedRegions = new Set(current.missedRegions);
      if (missedRegions.has(region)) missedRegions.delete(region);
      else missedRegions.add(region);
      return { ...current, missedRegions };
    });
    setError(null);
  };

  const submit = async () => {
    if (!view || !draft || submitting) return;
    const incomplete = view.detections.some(
      (detection) => !draft.items[detection.detectionId]?.verdict,
    );
    if (incomplete) {
      setError("请完成所有已检测区域的判定");
      return;
    }
    const label1WithoutColor = view.detections.some(
      (detection) => detection.logicalRegion === "label1"
        && !draft.items[detection.detectionId]?.color,
    );
    if (label1WithoutColor) {
      setError("请为 label1 选择颜色");
      return;
    }
    if (view.detections.length === 0 && draft.missedRegions.size === 0) {
      setError("请至少选择一个漏检区域");
      return;
    }

    const payload: FeedbackUpdateRequest = {
      items: view.detections.map((detection) => {
        const item = draft.items[detection.detectionId];
        const result: FeedbackUpdateRequest["items"][number] = {
          detectionId: detection.detectionId,
          verdict: item.verdict!,
        };
        if (detection.logicalRegion === "label1") result.color = item.color;
        return result;
      }),
      missedRegions: view.missedRegionCandidates.filter(
        (region) => draft.missedRegions.has(region),
      ),
    };

    setSubmitting(true);
    setError(null);
    try {
      const updated = await client.updateImageFeedback(imageId, payload);
      setView(updated);
      setSaved(true);
      onSaved(updated);
      closeTimer.current = window.setTimeout(onClose, 650);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "反馈保存失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="feedback-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !submitting) onClose();
      }}
    >
      <section className="feedback-dialog" role="dialog" aria-modal="true" aria-labelledby="feedback-dialog-title">
        <header className="feedback-dialog__header">
          <div>
            <h2 id="feedback-dialog-title">检测结果反馈</h2>
            <p title={filename}>{filename}</p>
          </div>
          <button type="button" aria-label="关闭反馈" disabled={submitting} onClick={onClose}>
            <X size={20} />
          </button>
        </header>

        <div className="feedback-dialog__body">
          {loading ? <p className="feedback-dialog__loading">正在读取已有反馈…</p> : null}
          {!loading && !view ? (
            <div className="feedback-dialog__load-error">
              <p role="alert"><WarningCircle size={18} />{error ?? "反馈加载失败"}</p>
              <button className="button button--secondary" type="button" onClick={() => void load()}>
                重试加载反馈
              </button>
            </div>
          ) : null}

          {!loading && view && draft ? (
            <>
              <div className="feedback-dialog__section-heading">
                <h3>已检测区域</h3>
                <span>请按实际情况确认，不默认采用模型结果</span>
              </div>
              {view.detections.length ? (
                <div className="feedback-regions">
                  {view.detections.map((detection) => {
                    const item = draft.items[detection.detectionId] ?? {};
                    return (
                      <article className="feedback-region" key={detection.detectionId}>
                        <div className="feedback-region__name">
                          <strong>{detection.logicalRegion}</strong>
                          {detection.logicalRegion === "label1" ? (
                            <span>实际检测：{detection.regionLabel}</span>
                          ) : null}
                        </div>
                        <fieldset aria-label={`${detection.logicalRegion} 判定`}>
                          <legend>实际结果</legend>
                          {(["OK", "NG"] as FeedbackVerdict[]).map((verdict) => (
                            <label key={verdict} className={`feedback-choice feedback-choice--${verdict.toLowerCase()}`}>
                              <input
                                type="radio"
                                name={`verdict-${detection.detectionId}`}
                                aria-label={`${detection.logicalRegion} ${verdict}`}
                                checked={item.verdict === verdict}
                                onChange={() => setItem(detection.detectionId, { verdict })}
                              />
                              {verdict}
                            </label>
                          ))}
                        </fieldset>
                        {detection.logicalRegion === "label1" ? (
                          <fieldset aria-label="label1 颜色">
                            <legend>实际颜色</legend>
                            {COLORS.map((color) => (
                              <label key={color} className="feedback-choice">
                                <input
                                  type="radio"
                                  name={`color-${detection.detectionId}`}
                                  aria-label={`label1 颜色 ${color}`}
                                  checked={item.color === color}
                                  onChange={() => setItem(detection.detectionId, { color })}
                                />
                                {color}
                              </label>
                            ))}
                          </fieldset>
                        ) : null}
                      </article>
                    );
                  })}
                </div>
              ) : (
                <p className="feedback-dialog__empty">本图没有检测到任何区域，请在下方标记漏检区域。</p>
              )}

              <div className="feedback-dialog__section-heading feedback-dialog__section-heading--missed">
                <h3>漏检区域</h3>
                <span>仅列出本图未检测到的区域</span>
              </div>
              {view.missedRegionCandidates.length ? (
                <div className="feedback-missed-regions">
                  {view.missedRegionCandidates.map((region) => (
                    <label key={region} className="feedback-missed-choice">
                      <input
                        type="checkbox"
                        aria-label={`${region} 漏检`}
                        checked={draft.missedRegions.has(region)}
                        onChange={() => toggleMissed(region)}
                      />
                      <span>{region}</span>
                    </label>
                  ))}
                </div>
              ) : <p className="feedback-dialog__empty">所有区域均已检测，无可选漏检区域。</p>}

              {error ? <p className="feedback-dialog__error" role="alert">{error}</p> : null}
              {saved ? <p className="feedback-dialog__success" role="status">反馈已保存</p> : null}
            </>
          ) : null}
        </div>

        {view && draft ? (
          <footer className="feedback-dialog__actions">
            <button className="button button--secondary" type="button" disabled={submitting} onClick={onClose}>取消</button>
            <button className="button button--primary" type="button" disabled={submitting || saved} onClick={() => void submit()}>
              {submitting ? "正在提交…" : saved ? "已保存" : "提交反馈"}
            </button>
          </footer>
        ) : null}
      </section>
    </div>
  );
}
