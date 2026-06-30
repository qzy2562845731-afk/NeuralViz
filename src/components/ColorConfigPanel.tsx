import { useState, useRef, useEffect } from 'react';
import type { ChartColors } from '../hooks/useColorConfig';
import { PRESET_LABELS, PRESETS } from '../hooks/useColorConfig';

interface ColorConfigPanelProps {
  colors: ChartColors;
  onUpdateColor: (key: keyof ChartColors, value: string | string[]) => void;
  onRestoreDefaults: () => void;
  onApplyPreset: (preset: keyof typeof PRESET_LABELS) => void;
  currentPreset: keyof typeof PRESET_LABELS | null;
}

interface ColorItemProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  showPreview?: boolean;
}

function ColorItem({ label, value, onChange, showPreview = false }: ColorItemProps) {
  return (
    <div className="color-item">
      <span className="color-label">{label}</span>
      <div className="color-input-group">
        {showPreview && (
          <div className="color-preview" style={{ backgroundColor: value }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="2" x2="12" y2="22" />
              <polyline points="18 5 12 11 6 5" />
            </svg>
          </div>
        )}
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="color-picker"
        />
        <span className="color-value">{value.toUpperCase()}</span>
      </div>
    </div>
  );
}

interface PresetPreviewProps {
  presetKey: keyof typeof PRESET_LABELS;
  colors: ChartColors;
  isActive: boolean;
  onClick: () => void;
}

function PresetPreview({ presetKey, colors, isActive, onClick }: PresetPreviewProps) {
  return (
    <button
      className={`preset-preview ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      <div className="preset-colors">
        <div className="preset-color" style={{ backgroundColor: colors.trainLoss }} />
        <div className="preset-color" style={{ backgroundColor: colors.valLoss }} />
        <div className="preset-color" style={{ backgroundColor: colors.trainAccuracy }} />
        <div className="preset-color" style={{ backgroundColor: colors.valAccuracy }} />
        <div className="preset-color" style={{ backgroundColor: colors.learningRate }} />
      </div>
      <span className="preset-name">{PRESET_LABELS[presetKey]}</span>
      {isActive && (
        <span className="preset-active-badge">Active</span>
      )}
    </button>
  );
}

export function ColorConfigPanel({
  colors,
  onUpdateColor,
  onRestoreDefaults,
  onApplyPreset,
  currentPreset,
}: ColorConfigPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="color-config" ref={panelRef}>
      <button
        className={`color-toggle-btn ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="Color Settings"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="4" />
          <line x1="12" y1="2" x2="12" y2="6" />
          <line x1="12" y1="18" x2="12" y2="22" />
          <line x1="2" y1="12" x2="6" y2="12" />
          <line x1="18" y1="12" x2="22" y2="12" />
        </svg>
        <span className="color-toggle-label">Colors</span>
      </button>

      {isOpen && (
        <div className="color-panel">
          <div className="color-panel-header">
            <h3>Theme & Colors</h3>
            {currentPreset && (
              <span className="current-preset">
                Active: <strong>{PRESET_LABELS[currentPreset]}</strong>
              </span>
            )}
          </div>

          <div className="color-presets">
            <h4>Quick Themes</h4>
            <div className="presets-grid">
              {Object.entries(PRESET_LABELS).map(([key]) => {
                const presetKey = key as keyof typeof PRESET_LABELS;
                return (
                  <PresetPreview
                    key={key}
                    presetKey={presetKey}
                    colors={PRESETS[presetKey]}
                    isActive={currentPreset === key}
                    onClick={() => onApplyPreset(presetKey)}
                  />
                );
              })}
            </div>
          </div>

          <div className="color-sections">
            <div className="color-section">
              <h4>Training Metrics</h4>
              <ColorItem
                label="Train Loss"
                value={colors.trainLoss}
                onChange={(v) => onUpdateColor('trainLoss', v)}
                showPreview
              />
              <ColorItem
                label="Val Loss"
                value={colors.valLoss}
                onChange={(v) => onUpdateColor('valLoss', v)}
                showPreview
              />
              <ColorItem
                label="Train Accuracy"
                value={colors.trainAccuracy}
                onChange={(v) => onUpdateColor('trainAccuracy', v)}
                showPreview
              />
              <ColorItem
                label="Val Accuracy"
                value={colors.valAccuracy}
                onChange={(v) => onUpdateColor('valAccuracy', v)}
                showPreview
              />
              <ColorItem
                label="Learning Rate"
                value={colors.learningRate}
                onChange={(v) => onUpdateColor('learningRate', v)}
                showPreview
              />
            </div>

            <div className="color-section">
              <h4>Analysis Charts</h4>
              <ColorItem
                label="Gradient Norm"
                value={colors.gradientNorm}
                onChange={(v) => onUpdateColor('gradientNorm', v)}
                showPreview
              />
              <ColorItem
                label="Weight Norm"
                value={colors.weightNorm}
                onChange={(v) => onUpdateColor('weightNorm', v)}
                showPreview
              />
              <ColorItem
                label="Confusion Matrix"
                value={colors.confusionMatrix}
                onChange={(v) => onUpdateColor('confusionMatrix', v)}
                showPreview
              />
            </div>

            <div className="color-section">
              <h4>CNN & Activation</h4>
              <div className="color-item">
                <span className="color-label">CNN Layers</span>
                <div className="color-layer-inputs">
                  {colors.cnnLayers.map((color, idx) => (
                    <div key={idx} className="layer-color-wrapper">
                      <input
                        type="color"
                        value={color}
                        onChange={(e) => {
                          const newLayers = [...colors.cnnLayers];
                          newLayers[idx] = e.target.value;
                          onUpdateColor('cnnLayers', newLayers);
                        }}
                        className="color-picker small"
                        title={`Layer ${idx + 1}`}
                      />
                      <span className="layer-index">{idx + 1}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="color-item">
                <span className="color-label">Histogram Layers</span>
                <div className="color-layer-inputs">
                  {colors.activationHistogram.map((color, idx) => (
                    <div key={idx} className="layer-color-wrapper">
                      <input
                        type="color"
                        value={color}
                        onChange={(e) => {
                          const newLayers = [...colors.activationHistogram];
                          newLayers[idx] = e.target.value;
                          onUpdateColor('activationHistogram', newLayers);
                        }}
                        className="color-picker small"
                        title={`Layer ${idx + 1}`}
                      />
                      <span className="layer-index">{idx + 1}</span>
                    </div>
                  ))}
                </div>
              </div>
              <ColorItem
                label="Feature Maps"
                value={colors.featureMaps}
                onChange={(v) => onUpdateColor('featureMaps', v)}
                showPreview
              />
            </div>

            <div className="color-section">
              <h4>Status Colors</h4>
              <div className="status-colors-preview">
                <div className="status-color-item" style={{ '--status-color': colors.success } as React.CSSProperties}>
                  <span className="status-dot"></span>
                  <span className="status-text">Success</span>
                </div>
                <div className="status-color-item" style={{ '--status-color': colors.warning } as React.CSSProperties}>
                  <span className="status-dot"></span>
                  <span className="status-text">Warning</span>
                </div>
                <div className="status-color-item" style={{ '--status-color': colors.danger } as React.CSSProperties}>
                  <span className="status-dot"></span>
                  <span className="status-text">Danger</span>
                </div>
                <div className="status-color-item" style={{ '--status-color': colors.info } as React.CSSProperties}>
                  <span className="status-dot"></span>
                  <span className="status-text">Info</span>
                </div>
              </div>
              <ColorItem
                label="Success"
                value={colors.success}
                onChange={(v) => onUpdateColor('success', v)}
              />
              <ColorItem
                label="Warning"
                value={colors.warning}
                onChange={(v) => onUpdateColor('warning', v)}
              />
              <ColorItem
                label="Danger"
                value={colors.danger}
                onChange={(v) => onUpdateColor('danger', v)}
              />
              <ColorItem
                label="Info"
                value={colors.info}
                onChange={(v) => onUpdateColor('info', v)}
              />
            </div>

            <div className="color-section">
              <h4>UI Accents</h4>
              <ColorItem
                label="Primary"
                value={colors.primary}
                onChange={(v) => onUpdateColor('primary', v)}
                showPreview
              />
              <ColorItem
                label="Secondary"
                value={colors.secondary}
                onChange={(v) => onUpdateColor('secondary', v)}
                showPreview
              />
              <ColorItem
                label="Accent"
                value={colors.accent}
                onChange={(v) => onUpdateColor('accent', v)}
                showPreview
              />
            </div>
          </div>

          <div className="color-panel-footer">
            <button className="color-restore-btn" onClick={onRestoreDefaults}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
              Restore Defaults
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
