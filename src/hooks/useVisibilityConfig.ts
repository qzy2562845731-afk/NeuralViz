import { useState, useEffect, useCallback } from 'react';

export interface VisibilityConfig {
  trainingCurves: boolean;
  cnnViewer: boolean;
  featureMaps: boolean;
  modelMetrics: boolean;
  diagnosis: boolean;
  confusionMatrix: boolean;
  activationHistogram: boolean;
  gradientWeightChart: boolean;
}

const STORAGE_KEY = 'nn-visibility-config';

const DEFAULT_CONFIG: VisibilityConfig = {
  trainingCurves: true,
  cnnViewer: true,
  featureMaps: true,
  modelMetrics: true,
  diagnosis: true,
  confusionMatrix: true,
  activationHistogram: true,
  gradientWeightChart: true,
};

const MINIMAL_CONFIG: VisibilityConfig = {
  trainingCurves: true,
  cnnViewer: true,
  featureMaps: false,
  modelMetrics: false,
  diagnosis: false,
  confusionMatrix: false,
  activationHistogram: false,
  gradientWeightChart: false,
};

function loadFromStorage(): VisibilityConfig | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored) as VisibilityConfig;
    }
  } catch {
    // Ignore storage errors
  }
  return null;
}

function saveToStorage(config: VisibilityConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // Ignore storage errors
  }
}

export interface UseVisibilityConfigResult {
  config: VisibilityConfig;
  toggleSection: (section: keyof VisibilityConfig) => void;
  showAll: () => void;
  showMinimal: () => void;
  resetToDefault: () => void;
  isDefault: boolean;
  isMinimal: boolean;
  visibleCount: number;
}

export function useVisibilityConfig(): UseVisibilityConfigResult {
  const [config, setConfig] = useState<VisibilityConfig>(() => {
    return loadFromStorage() ?? DEFAULT_CONFIG;
  });

  useEffect(() => {
    saveToStorage(config);
  }, [config]);

  const toggleSection = useCallback((section: keyof VisibilityConfig) => {
    setConfig(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  }, []);

  const showAll = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
  }, []);

  const showMinimal = useCallback(() => {
    setConfig(MINIMAL_CONFIG);
  }, []);

  const resetToDefault = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
  }, []);

  const isDefault = JSON.stringify(config) === JSON.stringify(DEFAULT_CONFIG);
  const isMinimal = JSON.stringify(config) === JSON.stringify(MINIMAL_CONFIG);
  const visibleCount = Object.values(config).filter(Boolean).length;

  return {
    config,
    toggleSection,
    showAll,
    showMinimal,
    resetToDefault,
    isDefault,
    isMinimal,
    visibleCount,
  };
}
