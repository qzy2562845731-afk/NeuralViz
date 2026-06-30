import { useState, useEffect, useCallback } from 'react';
import type { AISettings, AIProvider } from '../types/ai';
import { DEFAULT_AI_SETTINGS, AI_PROVIDER_PRESETS } from '../types/ai';

const STORAGE_KEY = 'nn-ai-settings-v1';

function loadSettings(): AISettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as AISettings;
      return { ...DEFAULT_AI_SETTINGS, ...parsed };
    }
  } catch {
    // ignore
  }
  return DEFAULT_AI_SETTINGS;
}

function saveSettings(settings: AISettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // ignore
  }
}

export interface UseAISettingsResult {
  settings: AISettings;
  updateProvider: (provider: AIProvider) => void;
  updateApiKey: (apiKey: string) => void;
  updateEndpoint: (endpoint: string) => void;
  updateModel: (model: string) => void;
  updateTemperature: (temperature: number) => void;
  updateMaxTokens: (maxTokens: number) => void;
  toggleEnabled: () => void;
  reset: () => void;
  isConfigured: boolean;
}

export function useAISettings(): UseAISettingsResult {
  const [settings, setSettings] = useState<AISettings>(loadSettings);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const updateProvider = useCallback((provider: AIProvider) => {
    setSettings((prev) => ({
      ...prev,
      provider,
      apiEndpoint: AI_PROVIDER_PRESETS[provider].endpoint,
      model: AI_PROVIDER_PRESETS[provider].models[0],
    }));
  }, []);

  const updateApiKey = useCallback((apiKey: string) => {
    setSettings((prev) => ({ ...prev, apiKey }));
  }, []);

  const updateEndpoint = useCallback((apiEndpoint: string) => {
    setSettings((prev) => ({ ...prev, apiEndpoint }));
  }, []);

  const updateModel = useCallback((model: string) => {
    setSettings((prev) => ({ ...prev, model }));
  }, []);

  const updateTemperature = useCallback((temperature: number) => {
    setSettings((prev) => ({ ...prev, temperature }));
  }, []);

  const updateMaxTokens = useCallback((maxTokens: number) => {
    setSettings((prev) => ({ ...prev, maxTokens }));
  }, []);

  const toggleEnabled = useCallback(() => {
    setSettings((prev) => ({ ...prev, enabled: !prev.enabled }));
  }, []);

  const reset = useCallback(() => {
    setSettings(DEFAULT_AI_SETTINGS);
  }, []);

  const isConfigured = Boolean(
    settings.enabled && settings.apiKey && settings.apiEndpoint && settings.model
  );

  return {
    settings,
    updateProvider,
    updateApiKey,
    updateEndpoint,
    updateModel,
    updateTemperature,
    updateMaxTokens,
    toggleEnabled,
    reset,
    isConfigured,
  };
}
