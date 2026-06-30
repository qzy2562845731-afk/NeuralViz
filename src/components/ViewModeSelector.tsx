import { useState, useRef, useEffect } from 'react';
import type { PageViewMode } from '../hooks/useViewMode';
import { VIEW_MODE_LABELS } from '../hooks/useViewMode';

interface ViewModeSelectorProps {
  currentMode: PageViewMode;
  onModeChange: (mode: PageViewMode) => void;
}

export function ViewModeSelector({ currentMode, onModeChange }: ViewModeSelectorProps) {
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

  const modes: PageViewMode[] = ['2d-analysis', '3d-model', 'split-view'];

  return (
    <div className="view-mode-selector" ref={dropdownRef}>
      <button
        className={`view-mode-toggle ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="View Mode"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="12" y1="3" x2="12" y2="21" />
        </svg>
        <span className="view-mode-label">{VIEW_MODE_LABELS[currentMode]}</span>
      </button>

      {isOpen && (
        <div className="view-mode-dropdown">
          {modes.map((mode) => (
            <button
              key={mode}
              className={`view-mode-option ${currentMode === mode ? 'active' : ''}`}
              onClick={() => {
                onModeChange(mode);
                setIsOpen(false);
              }}
            >
              <span className="view-mode-option-icon">
                {mode === '2d-analysis' && (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <line x1="3" y1="12" x2="21" y2="12" />
                    <line x1="12" y1="3" x2="12" y2="21" />
                  </svg>
                )}
                {mode === '3d-model' && (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                )}
                {mode === 'split-view' && (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <line x1="12" y1="3" x2="12" y2="21" />
                    <rect x="3" y="3" width="9" height="18" rx="2" fill="currentColor" fillOpacity="0.1" />
                  </svg>
                )}
              </span>
              <span className="view-mode-option-label">{VIEW_MODE_LABELS[mode]}</span>
              {currentMode === mode && (
                <span className="view-mode-active-indicator">✓</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}