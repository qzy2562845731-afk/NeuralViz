import type { PlaybackSpeed } from '../types/training';
import type { VisibilityConfig } from '../hooks/useVisibilityConfig';
import type { ChartColors } from '../hooks/useColorConfig';
import type { PageViewMode } from '../hooks/useViewMode';
import { PRESET_LABELS } from '../hooks/useColorConfig';
import { LayoutControl } from './LayoutControl';
import { ColorConfigPanel } from './ColorConfigPanel';
import { DownloadButton } from './DownloadButton';
import { ViewModeSelector } from './ViewModeSelector';

interface HeaderProps {
  currentStep: number;
  maxStep: number;
  isPlaying: boolean;
  isCompleted: boolean;
  speed: PlaybackSpeed;
  visibilityConfig: VisibilityConfig;
  onToggleVisibility: (section: keyof VisibilityConfig) => void;
  onShowAll: () => void;
  onShowMinimal: () => void;
  onResetVisibility: () => void;
  isDefaultLayout: boolean;
  isMinimalLayout: boolean;
  visibleModuleCount: number;
  chartColors: ChartColors;
  onUpdateColor: (key: keyof ChartColors, value: string | string[]) => void;
  onRestoreColors: () => void;
  onApplyPreset: (preset: keyof typeof PRESET_LABELS) => void;
  currentPreset: keyof typeof PRESET_LABELS | null;
  viewMode: PageViewMode;
  onViewModeChange: (mode: PageViewMode) => void;
}

export function Header({
  currentStep,
  maxStep,
  isPlaying,
  isCompleted,
  speed,
  visibilityConfig,
  onToggleVisibility,
  onShowAll,
  onShowMinimal,
  onResetVisibility,
  isDefaultLayout,
  isMinimalLayout,
  visibleModuleCount,
  chartColors,
  onUpdateColor,
  onRestoreColors,
  onApplyPreset,
  currentPreset,
  viewMode,
  onViewModeChange,
}: HeaderProps) {
  const getStatusText = () => {
    if (isCompleted && !isPlaying) return 'Completed';
    return isPlaying ? 'Running' : 'Paused';
  };

  const getStatusClass = () => {
    if (isCompleted && !isPlaying) return 'completed';
    return isPlaying ? 'playing' : 'idle';
  };

  return (
    <header className="dashboard-header">
      <div className="flex items-center gap-4">
        <div className="header-logo">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="3" />
            <circle cx="12" cy="12" r="9" strokeDasharray="4 2" />
            <line x1="12" y1="1" x2="12" y2="4" />
            <line x1="12" y1="20" x2="12" y2="23" />
            <line x1="1" y1="12" x2="4" y2="12" />
            <line x1="20" y1="12" x2="23" y2="12" />
          </svg>
        </div>
        <div className="flex flex-col">
          <h1 className="text-base font-semibold tracking-tight text-foreground">
            Neural Training
          </h1>
          <span className="text-[11px] text-muted-foreground tracking-wide">
            Visualization Dashboard
          </span>
        </div>
      </div>

      <div className="header-step">
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Step</span>
        <span className="font-semibold">{currentStep}</span>
        <span className="text-muted-foreground">/</span>
        <span className="text-muted-foreground">{maxStep}</span>
      </div>

      <div className="flex items-center gap-3">
        <ViewModeSelector
          currentMode={viewMode}
          onModeChange={onViewModeChange}
        />

        <DownloadButton
          filename="nn-training"
          currentStep={currentStep}
          maxStep={maxStep}
          speed={speed}
          isPlaying={isPlaying}
          isCompleted={isCompleted}
        />

        {viewMode === '2d-analysis' && (
          <LayoutControl
            config={visibilityConfig}
            onToggle={onToggleVisibility}
            onShowAll={onShowAll}
            onShowMinimal={onShowMinimal}
            onReset={onResetVisibility}
            isDefault={isDefaultLayout}
            isMinimal={isMinimalLayout}
            visibleCount={visibleModuleCount}
          />
        )}

        <ColorConfigPanel
          colors={chartColors}
          onUpdateColor={onUpdateColor}
          onRestoreDefaults={onRestoreColors}
          onApplyPreset={onApplyPreset}
          currentPreset={currentPreset}
        />

        <div className="flex flex-col items-end">
          <span className="text-[9px] uppercase tracking-widest text-muted-foreground">Speed</span>
          <span className="font-mono text-sm font-semibold tabular-nums text-foreground">
            {speed}x
          </span>
        </div>

        <div className={`status-indicator ${getStatusClass()}`}>
          <span className="status-dot" />
          <span>{getStatusText()}</span>
        </div>
      </div>
    </header>
  );
}