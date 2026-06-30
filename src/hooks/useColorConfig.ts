import { useState, useEffect, useCallback } from 'react';

export interface ChartColors {
  // 训练指标
  trainLoss: string;
  valLoss: string;
  trainAccuracy: string;
  valAccuracy: string;
  learningRate: string;
  
  // 分析图表
  gradientNorm: string;
  weightNorm: string;
  confusionMatrix: string;
  
  // 可视化元素
  activationHistogram: string[];
  cnnLayers: string[];
  featureMaps: string;
  
  // 状态颜色
  success: string;
  warning: string;
  danger: string;
  info: string;
  
  // 背景和UI
  primary: string;
  secondary: string;
  accent: string;
}

export interface ColorConfigResult {
  colors: ChartColors;
  updateColor: (key: keyof ChartColors, value: string | string[]) => void;
  restoreDefaults: () => void;
  applyPreset: (preset: keyof typeof PRESETS) => void;
  presets: typeof PRESETS;
  currentPreset: keyof typeof PRESETS | null;
}

const STORAGE_KEY = 'nn-chart-colors';
const PRESET_KEY = 'nn-color-preset';

const RESEARCH_DARK: ChartColors = {
  // Training Metrics — restrained, distinct hues
  trainLoss: '#e879f9',
  valLoss: '#fbbf24',
  trainAccuracy: '#4ade80',
  valAccuracy: '#60a5fa',
  learningRate: '#f472b6',

  // Analysis Charts
  gradientNorm: '#fbbf24',
  weightNorm: '#c084fc',
  confusionMatrix: '#4ade80',

  // Visualization layers — avoid the cyan/blue/purple stack
  activationHistogram: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  cnnLayers: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  featureMaps: '#c084fc',

  // Status Colors
  success: '#4ade80',
  warning: '#fbbf24',
  danger: '#f87171',
  info: '#60a5fa',

  // UI Colors
  primary: '#4ade80',
  secondary: '#64748b',
  accent: '#e879f9',
};

const COOL_NEON: ChartColors = {
  // Training Metrics — teal-anchored, not cyan-purple
  trainLoss: '#2dd4bf',
  valLoss: '#facc15',
  trainAccuracy: '#22d3ee',
  valAccuracy: '#34d399',
  learningRate: '#e879f9',

  // Analysis Charts
  gradientNorm: '#facc15',
  weightNorm: '#a78bfa',
  confusionMatrix: '#34d399',

  // Visualization
  activationHistogram: ['#2dd4bf', '#34d399', '#facc15', '#e879f9'],
  cnnLayers: ['#2dd4bf', '#34d399', '#facc15', '#a78bfa'],
  featureMaps: '#2dd4bf',

  // Status Colors
  success: '#34d399',
  warning: '#facc15',
  danger: '#f87171',
  info: '#22d3ee',

  // UI Colors
  primary: '#2dd4bf',
  secondary: '#64748b',
  accent: '#e879f9',
};

const SOFT_CONTRAST: ChartColors = {
  // Training Metrics — higher-chroma pastels, readable on dark
  trainLoss: '#c4b5fd',
  valLoss: '#fde68a',
  trainAccuracy: '#a7f3d0',
  valAccuracy: '#bfdbfe',
  learningRate: '#fbcfe8',

  // Analysis Charts
  gradientNorm: '#fde68a',
  weightNorm: '#c4b5fd',
  confusionMatrix: '#a7f3d0',

  // Visualization
  activationHistogram: ['#bfdbfe', '#a7f3d0', '#fde68a', '#fbcfe8'],
  cnnLayers: ['#bfdbfe', '#a7f3d0', '#fde68a', '#c4b5fd'],
  featureMaps: '#bfdbfe',

  // Status Colors
  success: '#a7f3d0',
  warning: '#fde68a',
  danger: '#fecaca',
  info: '#bfdbfe',

  // UI Colors
  primary: '#bfdbfe',
  secondary: '#9ca3af',
  accent: '#fbcfe8',
};

const PRESETS = {
  'research-dark': RESEARCH_DARK,
  'cool-neon': COOL_NEON,
  'soft-contrast': SOFT_CONTRAST,
};

type PresetKey = keyof typeof PRESETS;

const PRESET_LABELS: Record<PresetKey, string> = {
  'research-dark': 'Research Dark',
  'cool-neon': 'Cool Neon',
  'soft-contrast': 'Soft Contrast',
};

function loadFromStorage(): ChartColors | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored) as ChartColors;
    }
  } catch {
    // Ignore storage errors
  }
  return null;
}

function saveToStorage(colors: ChartColors): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(colors));
  } catch {
    // Ignore storage errors
  }
}

function loadPresetFromStorage(): PresetKey | null {
  try {
    const stored = localStorage.getItem(PRESET_KEY);
    if (stored && stored in PRESETS) {
      return stored as PresetKey;
    }
  } catch {
    // Ignore storage errors
  }
  return null;
}

function savePresetToStorage(preset: PresetKey): void {
  try {
    localStorage.setItem(PRESET_KEY, preset);
  } catch {
    // Ignore storage errors
  }
}

function calculateContrast(color1: string, color2: string): number {
  const hex = (c: string) => parseInt(c, 16);
  const r1 = hex(color1.slice(1, 3)) / 255;
  const g1 = hex(color1.slice(3, 5)) / 255;
  const b1 = hex(color1.slice(5, 7)) / 255;
  const r2 = hex(color2.slice(1, 3)) / 255;
  const g2 = hex(color2.slice(3, 5)) / 255;
  const b2 = hex(color2.slice(5, 7)) / 255;
  
  const l1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
  const l2 = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2;
  
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  
  return (lighter + 0.05) / (darker + 0.05);
}

function ensureReadableContrast(color: string, backgroundColor: string = '#0a0a14'): string {
  const contrast = calculateContrast(color, backgroundColor);
  if (contrast >= 4.5) {
    return color;
  }
  
  // Increase brightness until readable
  const hex = (c: string) => parseInt(c, 16);
  const r = Math.min(255, hex(color.slice(1, 3)) + 40);
  const g = Math.min(255, hex(color.slice(3, 5)) + 40);
  const b = Math.min(255, hex(color.slice(5, 7)) + 40);
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

export function useColorConfig(): ColorConfigResult {
  const [colors, setColors] = useState<ChartColors>(() => {
    const stored = loadFromStorage();
    if (stored) {
      return stored;
    }
    return RESEARCH_DARK;
  });

  const [currentPreset, setCurrentPreset] = useState<PresetKey | null>(() => {
    return loadPresetFromStorage();
  });

  useEffect(() => {
    saveToStorage(colors);
  }, [colors]);

  const updateColor = useCallback((key: keyof ChartColors, value: string | string[]) => {
    setColors(prev => {
      const newValue = Array.isArray(value)
        ? value.map(v => ensureReadableContrast(v))
        : ensureReadableContrast(value);
      
      return {
        ...prev,
        [key]: newValue,
      };
    });
    setCurrentPreset(null);
  }, []);

  const restoreDefaults = useCallback(() => {
    setColors(RESEARCH_DARK);
    setCurrentPreset('research-dark');
    savePresetToStorage('research-dark');
  }, []);

  const applyPreset = useCallback((preset: PresetKey) => {
    setColors(PRESETS[preset]);
    setCurrentPreset(preset);
    savePresetToStorage(preset);
  }, []);

  return {
    colors,
    updateColor,
    restoreDefaults,
    applyPreset,
    presets: PRESETS,
    currentPreset,
  };
}

export { PRESET_LABELS, PRESETS };
