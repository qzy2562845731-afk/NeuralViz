import { useState, useCallback } from 'react';

export type PageViewMode = '2d-analysis' | '3d-model' | 'split-view';

export interface PageViewModeConfig {
  mode: PageViewMode;
  setMode: (mode: PageViewMode) => void;
}

const STORAGE_KEY = 'nn-view-mode';

const VIEW_MODE_LABELS: Record<PageViewMode, string> = {
  '2d-analysis': '2D Analysis',
  '3d-model': '3D Model',
  'split-view': 'Split View',
};

function loadFromStorage(): PageViewMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && (stored === '2d-analysis' || stored === '3d-model' || stored === 'split-view')) {
      return stored as PageViewMode;
    }
  } catch {
    // Ignore storage errors
  }
  return '2d-analysis';
}

function saveToStorage(mode: PageViewMode): void {
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // Ignore storage errors
  }
}

export function useViewMode(): PageViewModeConfig {
  const [mode, setModeState] = useState<PageViewMode>(() => loadFromStorage());

  const setMode = useCallback((newMode: PageViewMode) => {
    setModeState(newMode);
    saveToStorage(newMode);
  }, []);

  return {
    mode,
    setMode,
  };
}

export { VIEW_MODE_LABELS };