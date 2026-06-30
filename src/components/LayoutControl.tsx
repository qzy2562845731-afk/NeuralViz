import { useState, useRef, useEffect } from 'react';
import type { VisibilityConfig } from '../hooks/useVisibilityConfig';

interface SectionItem {
  key: keyof VisibilityConfig;
  label: string;
  icon: React.ReactNode;
}

interface LayoutControlProps {
  config: VisibilityConfig;
  onToggle: (section: keyof VisibilityConfig) => void;
  onShowAll: () => void;
  onShowMinimal: () => void;
  onReset: () => void;
  isDefault: boolean;
  isMinimal: boolean;
  visibleCount: number;
}

const SECTIONS: SectionItem[] = [
  {
    key: 'trainingCurves',
    label: 'Training Curves',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    key: 'cnnViewer',
    label: 'CNN Viewer',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    key: 'featureMaps',
    label: 'Feature Maps',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <rect x="7" y="7" width="3" height="3" />
        <rect x="14" y="7" width="3" height="3" />
        <rect x="7" y="14" width="3" height="3" />
        <rect x="14" y="14" width="3" height="3" />
      </svg>
    ),
  },
  {
    key: 'modelMetrics',
    label: 'Model Metrics',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 20V10" />
        <path d="M18 20V4" />
        <path d="M6 20v-4" />
      </svg>
    ),
  },
  {
    key: 'diagnosis',
    label: 'Diagnosis',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
  },
  {
    key: 'confusionMatrix',
    label: 'Confusion Matrix',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <line x1="3" y1="9" x2="21" y2="9" />
        <line x1="3" y1="15" x2="21" y2="15" />
        <line x1="9" y1="3" x2="9" y2="21" />
        <line x1="15" y1="3" x2="15" y2="21" />
      </svg>
    ),
  },
  {
    key: 'activationHistogram',
    label: 'Activation Histogram',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
  {
    key: 'gradientWeightChart',
    label: 'Gradient/Weight',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      </svg>
    ),
  },
];

export function LayoutControl({
  config,
  onToggle,
  onShowAll,
  onShowMinimal,
  onReset,
  isDefault,
  isMinimal,
  visibleCount,
}: LayoutControlProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div className="layout-control" ref={dropdownRef}>
      <button
        className={`layout-toggle-btn ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="View / Layout"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="7" height="9" rx="1" />
          <rect x="14" y="3" width="7" height="5" rx="1" />
          <rect x="14" y="12" width="7" height="9" rx="1" />
          <rect x="3" y="16" width="7" height="5" rx="1" />
        </svg>
        <span className="layout-toggle-label">View</span>
        <span className="layout-count">{visibleCount}/8</span>
      </button>

      {isOpen && (
        <div className="layout-dropdown" onKeyDown={handleKeyDown}>
          <div className="layout-dropdown-header">
            <h3>Module Visibility</h3>
            <span className="layout-info">{visibleCount} modules visible</span>
          </div>

          <div className="layout-sections">
            {SECTIONS.map((section) => (
              <label key={section.key} className="layout-section-item">
                <span className="section-icon">{section.icon}</span>
                <span className="section-label">{section.label}</span>
                <button
                  className={`section-toggle ${config[section.key] ? 'on' : 'off'}`}
                  onClick={() => onToggle(section.key)}
                  type="button"
                >
                  <span className="toggle-track">
                    <span className="toggle-thumb" />
                  </span>
                </button>
              </label>
            ))}
          </div>

          <div className="layout-actions">
            <button
              className={`layout-action-btn ${isDefault ? 'active' : ''}`}
              onClick={() => {
                onShowAll();
              }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
              </svg>
              Show All
            </button>
            <button
              className={`layout-action-btn ${isMinimal ? 'active' : ''}`}
              onClick={() => {
                onShowMinimal();
              }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
              Minimal
            </button>
            <button
              className="layout-action-btn reset"
              onClick={() => {
                onReset();
              }}
              disabled={isDefault}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
